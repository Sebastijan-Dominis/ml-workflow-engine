"""Training script entry point and helpers.

This module provides a small CLI entrypoint to run model training for
different tasks and algorithms. It exposes helper functions to parse
command-line arguments, load YAML configuration files, validate them
against the project's Pydantic schema, and run the appropriate trainer
implementation. After training, the resulting pipeline and metadata are
persisted and the global models registry is updated.

Typical usage:
    python -m ml.training.training.train \\
        --problem cancellation --segment global --version v1 \\
        --experiment-id 20260206_154343_2f5c2000

The ``--experiment-id`` flag must point to an existing experiment
directory created by a prior search run.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sklearn.pipeline import Pipeline

from ml.cli.error_handling import resolve_exit_code
from ml.config.hashing import add_config_hash
from ml.config.loader import load_and_validate_config
from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import PipelineContractError
from ml.logging_config import add_file_handler, bootstrap_logging
from ml.registry.allowed_models_registry import AllowedModels
from ml.registry.hash_registry import hash_artifact
from ml.runners.training.constants.output import TRAIN_OUTPUT
from ml.runners.training.persistence.artifacts.save_model import save_model
from ml.runners.training.persistence.artifacts.save_pipeline import \
    save_pipeline
from ml.runners.training.persistence.run_info.persist_training_run import \
    persist_training_run
from ml.runners.training.trainers.base import Trainer
from ml.runners.training.utils.get_trainer import get_trainer
from ml.runners.training.utils.logical_config_checks.validate_logical_config import \
    validate_logical_config
from ml.utils.experiments.lineage_integrity.validate_lineage_integrity import \
    validate_lineage_integrity
from ml.utils.experiments.logical_config.validate_pipeline_cfg import \
    validate_pipeline_cfg
from ml.utils.experiments.reproducibility.validate_reproducibility import \
    validate_reproducibility
from ml.utils.experiments.snapshot_path import get_snapshot_path

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model.")

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
        "--logging-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    return parser.parse_args()

def main() -> int:
    args: argparse.Namespace
    model_cfg: TrainModelConfig
    trainer: Trainer
    output: TRAIN_OUTPUT
    model: AllowedModels
    pipeline: Pipeline | None
    feature_lineage: list[dict]
    metrics: dict[str, float]
    pipeline_cfg_hash: str | None
    model_path: Path
    pipeline_path: Path | None

    args = parse_args()

    start_time = time.perf_counter()
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)

    bootstrap_logging(level=log_level)

    experiment_parent_dir = Path("experiments") / args.problem / args.segment / args.version
    experiment_dir = get_snapshot_path(args.experiment_id, experiment_parent_dir)
    search_dir = experiment_dir / "search"

    train_run_id = f"{timestamp}_{uuid4().hex[:8]}"
    train_run_dir = experiment_dir / "training" / train_run_id
    train_run_dir.mkdir(parents=True, exist_ok=False)

    add_file_handler(train_run_dir / f"train.log", level=log_level)

    try:
        model_cfg = load_and_validate_config(
            Path(f"configs/train/{args.problem}/{args.segment}/{args.version}.yaml"),
            cfg_type="train",
            env=args.env,
            search_dir=search_dir,
        )

        model_cfg = add_config_hash(model_cfg)

        validate_lineage_integrity(search_dir)
        validate_reproducibility(search_dir / "runtime.json")
        validate_logical_config(model_cfg, search_dir)
        validate_pipeline_cfg(search_dir / "metadata.json", model_cfg)

        algorithm = model_cfg.algorithm.value.lower()

        trainer = get_trainer(algorithm)

        logger.info(f"Starting training for problem={args.problem} segment={args.segment} version={args.version} using algorithm={algorithm}.")
        output = trainer.train(model_cfg, args.strict)

        model = output.model
        model_path = save_model(model, train_run_dir)
        model_hash = hash_artifact(model_path)

        pipeline = None
        pipeline_path = None
        pipeline_hash = None
        pipeline_cfg_hash = None
        if output.pipeline is not None:
            if output.pipeline_cfg_hash is None:
                msg = "Pipeline config hash is missing in the trainer output, but a pipeline object is present. This is unexpected as the pipeline config hash is needed for reproducibility tracking. Please ensure that the trainer implementation returns a pipeline config hash when a pipeline is returned."
                logger.error(msg)
                raise PipelineContractError(msg)
            
            pipeline = output.pipeline
            pipeline_path = save_pipeline(pipeline, train_run_dir)
            pipeline_hash = hash_artifact(pipeline_path)
            pipeline_cfg_hash = output.pipeline_cfg_hash
        

        feature_lineage = output.lineage
        metrics = output.metrics

        persist_training_run(
            model_cfg, 
            feature_lineage=feature_lineage, 
            start_time=start_time, 
            train_run_id=train_run_id, 
            experiment_dir=experiment_dir, 
            train_run_dir=train_run_dir, 
            metrics=metrics, 
            model_hash=model_hash, 
            pipeline_hash=pipeline_hash, 
            model_path=model_path, 
            pipeline_path=pipeline_path, 
            pipeline_cfg_hash=pipeline_cfg_hash, 
            timestamp=timestamp
        )

        logger.info(
            "Training completed | problem=%s segment=%s version=%s experiment_id=%s",
            args.problem,
            args.segment,
            args.version,
            args.experiment_id,
        )

        return 0

    except Exception as e:
        logger.exception("An error occurred during training.")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())