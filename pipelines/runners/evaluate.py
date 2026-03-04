"""Evaluation runner CLI.

This module provides a small command-line entrypoint to evaluate a trained
model. It validates the model configuration, dispatches to task-specific
evaluators and persists evaluation results via updater functions.

Typical usage example:
    python -m pipelines.runners.evaluate \\
        --problem cancellation --segment global --version v1 \\
        --experiment-id 20260206_154343_2f5c2000 \\
        --train-id 20260206_162120_sdg42000 \

The module exposes helper functions used by the CLI and a `main()` function
which orchestrates the complete evaluation flow.
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
from ml.config.schemas.model_cfg import TrainModelConfig
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.formatting.str_to_bool import str_to_bool
from ml.logging_config import add_file_handler, bootstrap_logging
from ml.modeling.models.artifacts import Artifacts
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.runners.evaluation.constants.output import EvaluateOutput
from ml.runners.evaluation.evaluators.base import Evaluator
from ml.runners.evaluation.models.predictions import PredictionArtifacts
from ml.runners.evaluation.persistence.persist_evaluation_run import persist_evaluation_run
from ml.runners.evaluation.utils.get_evaluator import get_evaluator
from ml.runners.shared.lineage.validate_lineage_integrity import validate_lineage_integrity
from ml.runners.shared.logical_config.validate_model_and_pipeline import validate_model_and_pipeline
from ml.runners.shared.logical_config.validate_pipeline_cfg import validate_pipeline_cfg
from ml.runners.shared.logical_config.validate_threshold import validate_threshold
from ml.runners.shared.reproducibility.validate_reproducibility import validate_reproducibility
from ml.types import LatestSnapshot
from ml.utils.snapshots.snapshot_path import get_snapshot_path

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a model.")

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

    return parser.parse_args()

def main() -> int:
    """Run model evaluation and persist evaluation artifacts.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Exceptions are converted to process exit codes; the function is designed
        as a CLI boundary and does not propagate failures upward.

    Side Effects:
        Creates evaluation run directories, writes predictions/metrics artifacts,
        metadata, runtime snapshots, and evaluation logs.

    Examples:
        python pipelines/runners/evaluate.py --problem cancellation --segment global --version v1 --experiment-id latest --train-id latest
    """
    args: argparse.Namespace
    model_cfg: TrainModelConfig
    pipeline_cfg_hash: str
    artifacts: Artifacts
    best_threshold: float | None
    evaluator: Evaluator
    output: EvaluateOutput
    metrics: dict[str, dict[str, float]]
    prediction_dfs: PredictionArtifacts
    feature_lineage: list[FeatureLineage]

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

    eval_run_id = f"{timestamp}_{uuid4().hex[:8]}"
    eval_run_dir = experiment_dir / "evaluation" / eval_run_id
    eval_run_dir.mkdir(parents=True, exist_ok=False)

    add_file_handler(eval_run_dir / "evaluation.log", level=log_level)

    try:
        model_cfg = load_and_validate_config(
            Path(f"configs/train/{args.problem}/{args.segment}/{args.version}.yaml"),
            cfg_type="train",
            env=args.env,
            search_dir=search_dir,
        )

        model_cfg = add_config_hash(model_cfg)

        validate_lineage_integrity(train_dir, model_cfg)
        validate_reproducibility(train_dir / "runtime.json")
        pipeline_cfg_hash = validate_pipeline_cfg(train_dir / "metadata.json", model_cfg)
        artifacts = validate_model_and_pipeline(train_dir)
        best_threshold = validate_threshold(model_cfg.task, train_dir / "metrics.json")

        key = model_cfg.task.type.lower()
        evaluator = get_evaluator(key)

        logger.info("Starting evaluation using experiment_id = %s and train_id = %s.", experiment_dir.name, train_run_id)
        output = evaluator.evaluate(model_cfg=model_cfg, strict=args.strict, best_threshold=best_threshold, train_dir=train_dir)
        logger.info("Evaluation completed. Persisting evaluation run...")

        metrics = output.metrics
        prediction_dfs = output.prediction_dfs
        feature_lineage = output.lineage

        persist_evaluation_run(
            model_cfg,
            eval_run_id=eval_run_id,
            train_run_id=train_run_id,
            experiment_dir=experiment_dir,
            eval_run_dir=eval_run_dir,
            metrics=metrics,
            prediction_dfs=prediction_dfs,
            feature_lineage=feature_lineage,
            start_time=start_time,
            timestamp=timestamp,
            artifacts=artifacts,
            pipeline_cfg_hash=pipeline_cfg_hash
        )

        logger.info("Evaluation run successfully persisted.")

        return 0

    except Exception as e:
        logger.exception("An error occurred during evaluation.")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())
