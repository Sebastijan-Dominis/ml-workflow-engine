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
from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import PipelineContractError
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.formatting.str_to_bool import str_to_bool
from ml.logging_config import add_file_handler, bootstrap_logging
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.runners.shared.lineage.validate_lineage_integrity import validate_lineage_integrity
from ml.runners.shared.logical_config.validate_pipeline_cfg import validate_pipeline_cfg
from ml.runners.shared.reproducibility.validate_reproducibility import validate_reproducibility
from ml.runners.training.constants.output import TrainOutput
from ml.runners.training.persistence.artifacts.save_model import save_model
from ml.runners.training.persistence.artifacts.save_pipeline import save_pipeline
from ml.runners.training.persistence.run_info.persist_training_run import persist_training_run
from ml.runners.training.trainers.base import Trainer
from ml.runners.training.utils.get_trainer import get_trainer
from ml.runners.training.utils.logical_config_checks.validate_logical_config import (
    validate_logical_config,
)
from ml.search.utils.failure_management.delete_failure_management_folder import (
    delete_failure_management_folder,
)
from ml.types import AllowedModels, LatestSnapshot
from ml.utils.hashing.service import hash_artifact
from ml.utils.snapshots.snapshot_path import get_snapshot_path
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
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
        "--train-run-id",
        type=str,
        default=None,
        help="Train run ID to use for this run (default: None, which generates a new unique train run ID). If provided, it should be in the format 'timestamp_randomstring', e.g., '20240101T120000_abcdef12'."
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
        "--logging-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    parser.add_argument(
        "--clean_up-failure-management",
        type=str_to_bool,
        default=True,
        help="Whether to clean up failure management folder after successful run (default: True)"
    )

    parser.add_argument(
        "--overwrite-existing",
        type=str_to_bool,
        default=False,
        help="Whether to overwrite existing train run data if the train run ID already exists (default: False). If False and files (other than train.log) already exist within the train run, the script will raise an error to prevent accidental data loss."
    )

    return parser.parse_args()

def main() -> int:
    """Run training, persist artifacts, and register run metadata.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Exceptions are converted to process exit codes; the function is designed
        as a CLI boundary and does not propagate failures upward.

    Side Effects:
        Creates/updates training run directories, writes logs/artifacts/metadata,
        and may delete failure-management folders on success.

    Examples:
        python pipelines/runners/train.py --problem cancellation --segment global --version v1 --experiment-id latest
    """
    args: argparse.Namespace
    model_cfg: TrainModelConfig
    trainer: Trainer
    output: TrainOutput
    model: AllowedModels
    pipeline: Pipeline | None
    feature_lineage: list[FeatureLineage]
    metrics: dict[str, float]
    pipeline_cfg_hash: str | None
    model_path: Path
    pipeline_path: Path | None

    args = parse_args()

    start_time = time.perf_counter()
    timestamp = iso_no_colon(datetime.now())

    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)

    bootstrap_logging(level=log_level)

    try:
        experiment_parent_dir = Path("experiments") / args.problem / args.segment / args.version
        experiment_dir = get_snapshot_path(args.experiment_id, experiment_parent_dir)
        print(f"Using experiment directory: {experiment_dir}\n")
        search_dir = experiment_dir / "search"
    except Exception as e:
        logger.exception("Failed to get experiment directory")
        return resolve_exit_code(e)

    train_run_id = args.train_run_id if args.train_run_id else f"{timestamp}_{uuid4().hex[:8]}"
    train_run_dir = experiment_dir / "training" / train_run_id

    if not train_run_dir.exists() and args.train_run_id:
        msg = f"Train run directory {train_run_dir} does not exist for provided train run ID {args.train_run_id}."
        logger.error(msg)
        return 1

    if train_run_dir.exists():
        add_file_handler(train_run_dir / "train.log", level=log_level)
        existing_files = [f.name for f in train_run_dir.iterdir() if f.is_file() and f.name != "train.log"]
        if existing_files and not args.overwrite_existing:
            msg = f"Train run directory {train_run_dir} already exists and contains files: {', '.join(existing_files)}. To prevent accidental data loss, the script will not overwrite existing train run data. To run training with this train run ID, please delete the existing files (if deemed appropriate), or set --overwrite-existing to True (if you want to overwrite the existing data)."
            logger.error(msg)
            return 1
    else:
        train_run_dir.mkdir(parents=True, exist_ok=False)
        add_file_handler(train_run_dir / "train.log", level=log_level)

    failure_management_dir = Path("failure_management") / experiment_dir.name / "training" / train_run_id
    failure_management_dir.mkdir(parents=True, exist_ok=True)

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

        logger.info(f"Starting training using experiment_id = {experiment_dir.name}.")
        output = trainer.train(
            model_cfg,
            strict=args.strict,
            failure_management_dir=failure_management_dir,
            search_dir=search_dir
        )
        logger.info("Training completed. Persisting training run...")

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

        logger.info("Training run successfully persisted.")

        delete_failure_management_folder(
            folder_path=failure_management_dir,
            cleanup=args.clean_up_failure_management,
            stage="train"
        )

        return 0

    except Exception as e:
        logger.exception("An error occurred during training.")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())
