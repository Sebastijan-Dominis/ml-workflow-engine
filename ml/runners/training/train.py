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

# General imports
import logging
logger = logging.getLogger(__name__)
import argparse
import sys
import yaml
from pathlib import Path
from typing import Any, Protocol

from ml.utils import load_model_specs, validate_model_specs

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.registry.train_registry import TRAIN_REGISTRY
from ml.runners.training.utils import load_train_and_val_data
from ml.runners.training.persistence.save_model import save_model
from ml.runners.training.persistence.save_pipeline import save_pipeline
from ml.runners.training.persistence.save_metadata import save_metadata
from ml.runners.training.persistence.update_general_config import update_general_config
from ml.logging_config import setup_logging
from ml.cli.error_handling import resolve_exit_code

class Trainer(Protocol):
    """
    Trainer interface.

    Returns:
        dict with keys:
        - best_params
        - phases
    """
    def train(self, model_cfg: TrainModelConfig) -> dict[str, Any]: ...

def parse_args() -> argparse.Namespace:
    try:
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
            "--experiment-id",
            type=str,
            required=True,
            help="Experiment id (directory name under experiments/{problem}/{segment}/{version})"
        )

        parser.add_argument(
            "--logging-level",
            type=str,
            default="INFO",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
        )

        return parser.parse_args()
        
    except Exception:
        logger.exception("Failed to parse arguments.")
        raise

def load_train_configs(problem, segment, version) -> dict:
    config_path = Path(f"configs/train/{problem}/{segment}/{version}.yaml")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception:
        logger.exception(f"Failed to load training configuration from {config_path}.")
        raise

def main() -> int:
    """Entrypoint to run model training according to a YAML configuration.

    The function performs the following high-level steps:
    1. Set up logging inside the experiment directory.
    2. Parse CLI arguments to obtain the config file name.
    3. Load and validate the YAML configuration.
    4. Select the appropriate trainer implementation based on the
       ``task`` and ``model.model_class`` specified in the config.
    5. Execute training, persist the resulting pipeline and metadata,
       and update the global models registry.

    Returns:
        0 on success, non-zero exit code on failure.
    """
    args: argparse.Namespace
    model_cfg: TrainModelConfig

    args = parse_args()

    experiment_dir = Path("experiments") / args.problem / args.segment / args.version / args.experiment_id
    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    setup_logging(experiment_dir / "train.log", level=log_level)

    try:
        cfg_model_specs_raw = load_model_specs(args.problem, args.segment, args.version, logger)

        cfg_model_specs = validate_model_specs(cfg_model_specs_raw, logger)

        cfg_train = load_train_configs(args.problem, args.segment, args.version)
        algorithm = cfg_model_specs["algorithm"]

        key = algorithm.lower()
        trainer = TRAIN_REGISTRY.get(key)

        if trainer:
            model, pipeline = trainer(cfg_model_specs, cfg_train)
        else:
            logger.error(f"No trainer found for algorithm '{algorithm}'.")
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        save_model(model, cfg_model_specs)
        save_pipeline(pipeline, cfg_model_specs)
        save_metadata(cfg_model_specs)

        ALGORITHMS_SUPPORTING_THRESHOLDS = ["catboost"]
        if algorithm.lower() in ALGORITHMS_SUPPORTING_THRESHOLDS:
            from ml.runners.training.utils import get_best_f1_thresh
            X_train, y_train, X_val, y_val = load_train_and_val_data(cfg_model_specs)

            best_threshold = get_best_f1_thresh(pipeline, X_val, y_val)
            update_general_config(cfg_model_specs, best_threshold)
        else:
            update_general_config(cfg_model_specs)

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