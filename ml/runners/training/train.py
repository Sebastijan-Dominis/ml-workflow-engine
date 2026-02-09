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

from ml.cli.error_handling import resolve_exit_code
from ml.config.hashing import add_config_hash
from ml.config.loader import load_and_validate_config
from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import ConfigError
from ml.logging_config import add_file_handler, bootstrap_logging
from ml.registry.train_registry import TRAIN_REGISTRY
from ml.runners.training.persistence.artifacts.save_model import save_model
from ml.runners.training.persistence.artifacts.save_pipeline import save_pipeline
from ml.runners.training.persistence.run_info.save_experiment import save_experiment
from ml.runners.training.utils.best_params_path import get_best_params_path
from ml.runners.training.utils.hashing.main import hash_artifact
from ml.runners.training.utils.logical_config_checks.validate_logical_config import validate_logical_config
from ml.utils.experiments.lineage_integrity.validate_lineage_integrity import validate_lineage_integrity
from ml.utils.experiments.logical_config.validate_pipeline_cfg import validate_pipeline_cfg
from ml.utils.experiments.reproducibility.validate_reproducibility import validate_reproducibility

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

    args = parse_args()

    start_time = time.perf_counter()
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)

    bootstrap_logging(level=log_level)

    experiment_parent_dir = Path("experiments") / args.problem / args.segment / args.version
    experiment_dir = get_best_params_path(args.experiment_id, experiment_parent_dir)

    train_run_id = f"{timestamp}_{uuid4().hex[:8]}"
    train_run_path = experiment_dir / "training" / train_run_id
    train_run_path.mkdir(parents=True, exist_ok=False)

    add_file_handler(train_run_path / f"train.log", level=log_level)

    try:
        model_cfg = load_and_validate_config(
            Path(f"configs/train/{args.problem}/{args.segment}/{args.version}.yaml"),
            cfg_type="train",
            env=args.env,
            experiment_dir=experiment_dir,
        )

        model_cfg = add_config_hash(model_cfg)

        validate_lineage_integrity(experiment_dir)
        validate_reproducibility(experiment_dir / "runtime.json")
        validate_logical_config(model_cfg, experiment_dir)
        validate_pipeline_cfg(experiment_dir / "experiment.json", model_cfg)

        algorithm = model_cfg.algorithm

        key = algorithm.value.lower()
        trainer = TRAIN_REGISTRY.get(key)

        if trainer:
            logger.info(f"Starting training for problem={args.problem} segment={args.segment} version={args.version} using algorithm={algorithm.value}.")
            model, pipeline, feature_lineage, metrics, pipeline_cfg_hash = trainer(model_cfg)
        else:
            msg = f"No trainer found for algorithm '{algorithm.value}'."
            logger.error(msg)
            raise ConfigError(msg)
        
        model_hash = hash_artifact(model)
        pipeline_hash = hash_artifact(pipeline)

        model_path = save_model(model, train_run_path)
        pipeline_path = save_pipeline(pipeline, train_run_path)
        save_experiment(model_cfg, feature_lineage=feature_lineage, start_time=start_time, train_run_id=train_run_id, experiment_dir=experiment_dir, metrics=metrics, model_hash=model_hash, pipeline_hash=pipeline_hash, model_path=model_path, pipeline_path=pipeline_path, pipeline_cfg_hash=pipeline_cfg_hash, timestamp=timestamp)


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