"""Evaluation runner CLI.

This module provides a small command-line entrypoint to evaluate a trained
model. It validates the model configuration, dispatches to task-specific
evaluators and persists evaluation results via updater functions.

Typical usage example:
    python -m ml.training.evaluation_scripts.evaluate \\
        --problem cancellation --segment global --version v1 \\
        --experiment-id 20260206_154343_2f5c2000

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

import pandas as pd

from ml.cli.error_handling import resolve_exit_code
from ml.config.hashing import add_config_hash
from ml.config.loader import load_and_validate_config
from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.logging_config import add_file_handler, bootstrap_logging
from ml.runners.evaluation.constants.output import EVALUATE_OUTPUT
from ml.runners.evaluation.evaluators.base import Evaluator
from ml.runners.evaluation.persistence.persist_evaluation_run import \
    persist_evaluation_run
from ml.runners.evaluation.utils.get_searcher import get_evaluator
from ml.utils.experiments.lineage_integrity.validate_lineage_integrity import \
    validate_lineage_integrity
from ml.utils.experiments.logical_config.validate_model_and_pipeline import \
    validate_model_and_pipeline
from ml.utils.experiments.logical_config.validate_pipeline_cfg import \
    validate_pipeline_cfg
from ml.utils.experiments.logical_config.validate_threshold import \
    validate_threshold
from ml.utils.experiments.reproducibility.validate_reproducibility import \
    validate_reproducibility
from ml.utils.experiments.snapshot_path import get_snapshot_path

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
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
        type=str,
        default="default",
        help="Environment to run the script in (dev/test/prod) (default: default) ~ none"
    )

    parser.add_argument(
        "--strict",
        type=bool,
        default=True,
        help="Whether to run in strict mode, which includes strict validation that may be computationally expensive (default: True)"
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default="latest",
        help="Experiment id (directory name under experiments/{problem}/{segment}/{version}); if not provided, defaults to 'latest' which picks the most recent experiment directory"
    )

    parser.add_argument(
        "--train-id",
        type=str,
        default="latest",
        help="Train id (directory name under experiments/{problem}/{segment}/{version}/{snapshot_id}/training); if not provided, defaults to 'latest' which picks the most recent training directory"
    )

    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    return parser.parse_args()

def main() -> int:
    args: argparse.Namespace
    model_cfg: TrainModelConfig
    pipeline_cfg_hash: str
    artifacts: dict[str, str]
    best_threshold: float | None
    evaluator: Evaluator
    output: EVALUATE_OUTPUT
    metrics: dict[str, dict[str, float]]
    prediction_dfs: dict[str, pd.DataFrame]
    feature_lineage: list[dict]

    args = parse_args()

    start_time = time.perf_counter()
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)

    bootstrap_logging(level=log_level)

    experiment_parent_dir = Path("experiments") / args.problem / args.segment / args.version
    experiment_dir = get_snapshot_path(args.experiment_id, experiment_parent_dir)
    search_dir = experiment_dir / "search"

    train_parent_dir = experiment_dir / "training"
    train_dir = get_snapshot_path(args.train_id, train_parent_dir)
    train_run_id = train_dir.name

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

        logger.info(
            "Starting evaluation | problem=%s segment=%s version=%s train_id=%s eval_id=%s",
            args.problem,
            args.segment,
            args.version,
            args.train_id,
            eval_run_id,
        )

        output = evaluator.evaluate(model_cfg=model_cfg, strict=args.strict, best_threshold=best_threshold, train_dir=train_dir)

        metrics = output.metrics
        prediction_dfs = output.prediction_dfs
        feature_lineage = output.lineage

        logger.info(
            "Evaluation completed | problem=%s segment=%s version=%s train_id=%s eval_id=%s",
            args.problem,
            args.segment,
            args.version,
            args.train_id,
            eval_run_id,
        )

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

        logger.info(
            "Evaluation results persisted | problem=%s segment=%s version=%s train_id=%s eval_id=%s",
            args.problem,
            args.segment,
            args.version,
            train_run_id,
            eval_run_id,
        )

        return 0

    except Exception as e:
        logger.exception("An error occurred during evaluation.")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())