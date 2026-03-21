"""Explainability runner CLI.

This module runs explainability for a selected trained run, validates lineage
and reproducibility constraints, and persists explainability outputs.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.cli.error_handling import resolve_exit_code
from ml.config.hashing import add_config_hash
from ml.config.loader import load_and_validate_config
from ml.exceptions import ConfigError
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.formatting.str_to_bool import str_to_bool
from ml.logging_config import add_file_handler, bootstrap_logging
from ml.runners.explainability.constants.output import ExplainabilityOutput
from ml.runners.explainability.persistence.persist_explainability_run import (
    persist_explainability_run,
)
from ml.runners.explainability.utils.get_explainer import get_explainer
from ml.runners.shared.lineage.validate_lineage_integrity import validate_lineage_integrity
from ml.runners.shared.logical_config.validate_model_and_pipeline import validate_model_and_pipeline
from ml.runners.shared.logical_config.validate_pipeline_cfg import validate_pipeline_cfg
from ml.runners.shared.reproducibility.validate_reproducibility import validate_reproducibility
from ml.types import LatestSnapshot
from ml.utils.snapshots.snapshot_path import get_snapshot_path

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for explainability execution.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Explain a model.")

    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Model problem, e.g., 'no_show'"
    )

    parser.add_argument(
        "--segment",
        type=str,
        required=True,
        help="Model segment name, e.g., 'city_hotel_online_ta'"
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version, e.g., 'v1'"
    )

    parser.add_argument(
        "--env",
        choices=["dev", "test", "prod", "default"],
        default="default",
        help="Environment to run the script in (dev/test/prod) (default: default) ~ none"
    )

    parser.add_argument(
        "--strict",
        type=str_to_bool,
        default=True,
        help="Whether to run in strict mode, which includes strict validation that may be computationally expensive (default: True)"
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default=LatestSnapshot.LATEST.value,
        help="Experiment id (directory name under experiments/{problem}/{segment}/{version}); if not provided, defaults to 'latest' which picks the most recent experiment directory"
    )

    parser.add_argument(
        "--train-id",
        type=str,
        default=LatestSnapshot.LATEST.value,
        help="Train id (directory name under experiments/{problem}/{segment}/{version}/{snapshot_id}/training); if not provided, defaults to 'latest' which picks the most recent training directory"
    )

    parser.add_argument(
        "--logging-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top features to include in the explainability output (will programmatically default to settings-specified value if not provided, but can be overridden with this flag)"
    )

    return parser.parse_args()

def main() -> int:
    """Execute explainability for a trained model run.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Exceptions are converted to process exit codes; the function is designed
        as a CLI boundary and does not propagate failures upward.

    Side Effects:
        Creates explainability run directories, writes explainability artifacts,
        metadata, runtime snapshots, and explainability logs.

    Examples:
        python pipelines/runners/explain.py --problem cancellation --segment global --version v1 --experiment-id latest --train-id latest --top-k 20
    """
    args: argparse.Namespace
    output: ExplainabilityOutput

    args = parse_args()

    start_time = time.perf_counter()
    timestamp = iso_no_colon(datetime.now())

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)

    bootstrap_logging(level=log_level)

    try:
        experiment_parent_dir = Path("experiments") / args.problem / args.segment / args.version
        experiment_dir = get_snapshot_path(args.experiment_id, experiment_parent_dir)
        print(f"Using experiment directory: {experiment_dir}")
        search_dir = experiment_dir / "search"

        train_parent_dir = experiment_dir / "training"
        train_dir = get_snapshot_path(args.train_id, train_parent_dir)
        print(f"Using training directory: {train_dir}")
        train_run_id = train_dir.name
    except Exception as e:
        logger.exception("Failed to get experiment or training snapshot path")
        return resolve_exit_code(e)

    explain_run_id = f"{timestamp}_{uuid4().hex[:8]}"
    explain_run_dir = experiment_dir / "explainability" / explain_run_id
    explain_run_dir.mkdir(parents=True, exist_ok=True)

    add_file_handler(explain_run_dir / "explainability.log", level=log_level)

    try:
        model_cfg = load_and_validate_config(
            Path(f"configs/train/{args.problem}/{args.segment}/{args.version}.yaml"),
            cfg_type="train",
            env=args.env,
            search_dir=search_dir,
        )

        if not model_cfg.explainability.enabled:
            msg = "Explainability is not enabled in the model configuration. Please enable it to run the explainability pipeline."
            logger.error(msg)
            raise ConfigError(msg)

        model_cfg = add_config_hash(model_cfg)

        validate_lineage_integrity(train_dir, model_cfg)
        validate_reproducibility(train_dir / "runtime.json")
        pipeline_cfg_hash = validate_pipeline_cfg(train_dir / "metadata.json", model_cfg)
        artifacts = validate_model_and_pipeline(train_dir)

        key = model_cfg.algorithm.name.lower()
        explainer = get_explainer(key)

        top_k = args.top_k if args.top_k is not None else model_cfg.explainability.top_k

        logger.info("Explaining the model using experiment_id = %s, train_id = %s, and top_k = %s.", experiment_dir.name, train_run_id, top_k)
        output = explainer.explain(model_cfg=model_cfg, train_dir=train_dir, top_k=top_k)
        logger.info("Explainability completed. Persisting explainability run...")

        explainability_metrics = output.explainability_metrics
        feature_lineage = output.feature_lineage


        persist_explainability_run(
            model_cfg=model_cfg,
            explain_run_id=explain_run_id,
            train_run_id=train_run_id,
            experiment_dir=experiment_dir,
            explain_run_dir=explain_run_dir,
            explainability_metrics=explainability_metrics,
            feature_lineage=feature_lineage,
            start_time=start_time,
            timestamp=timestamp,
            artifacts=artifacts,
            pipeline_cfg_hash=pipeline_cfg_hash,
            top_k=top_k
        )

        logger.info("Explainability run successfully persisted.")

        return 0

    except Exception as e:
        logger.exception("An error occurred during the explainability run.")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())
