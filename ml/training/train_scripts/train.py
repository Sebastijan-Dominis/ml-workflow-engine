"""Training script entry point and helpers.

This module provides a small CLI entrypoint to run model training for
different tasks and algorithms. It exposes helper functions to parse
command-line arguments, load YAML configuration files, validate them
against the project's Pydantic schema, and run the appropriate trainer
implementation. After training, the resulting pipeline and metadata are
persisted and the global models registry is updated.

Typical usage:
    python -m ml.training.train_scripts.train --name_version cancellation_global_v1

The script expects a YAML config at ``ml/training/train_configs/{name_version}.yaml``.
"""

# General imports
import logging
logger = logging.getLogger(__name__)
import argparse
import yaml
from pathlib import Path

from ml.utils import load_model_specs, validate_model_specs

# Specific training script imports
from ml.training.train_scripts.custom_training_scripts.train_catboost import train_catboost
from ml.training.train_scripts.utils import load_train_and_val_data

# Persistence imports
from ml.training.train_scripts.persistence.save_model import save_model
from ml.training.train_scripts.persistence.save_pipeline import save_pipeline
from ml.training.train_scripts.persistence.save_metadata import save_metadata
from ml.training.train_scripts.persistence.update_general_config import update_general_config

# from ml.training.train_scripts.persistence.update_general_config import (
#     update_general_config
# )

# Logger import
from ml.logging_config import setup_logging

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

def main() -> None:
    """Entrypoint to run model training according to a YAML configuration.

    The function performs the following high-level steps:
    1. Set up logging.
    2. Parse CLI arguments to obtain the config file name.
    3. Load and validate the YAML configuration.
    4. Select the appropriate trainer implementation based on the
       ``task`` and ``model.model_class`` specified in the config.
    5. Execute training, persist the resulting pipeline and metadata,
       and update the global models registry.

    Raises:
        ValueError: If no trainer is registered for the requested task/algorithm.
    """

    setup_logging() # Set up logging configuration

    args = parse_args() # Parse command-line arguments

    cfg_model_specs_raw = load_model_specs(args.problem, args.segment, args.version, logger) # Load raw YAML config

    cfg_model_specs = validate_model_specs(cfg_model_specs_raw, logger) # Validate config schema

    cfg_train = load_train_configs(args.problem, args.segment, args.version) # Load training-specific config
    algorithm = cfg_model_specs["algorithm"]

    # Trainer registry: extend this dict when adding new tasks/algorithms
    TRAINERS = {
        "catboost": train_catboost
    }

    key = algorithm.lower()
    trainer = TRAINERS.get(key)

    if trainer:
        model, pipeline = trainer(cfg_model_specs, cfg_train) # Execute training
    else:
        logger.error(f"No trainer found for algorithm '{algorithm}'.")
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    save_model(model, cfg_model_specs) # Persist model
    save_pipeline(pipeline, cfg_model_specs) # Persist pipeline
    save_metadata(cfg_model_specs) # Persist metadata

    ALGORITHMS_SUPPORTING_THRESHOLDS = ["catboost"]
    if algorithm.lower() in ALGORITHMS_SUPPORTING_THRESHOLDS:
        from ml.training.train_scripts.utils import get_best_f1_thresh
        X_train, y_train, X_val, y_val = load_train_and_val_data(cfg_model_specs)

        best_threshold = get_best_f1_thresh(pipeline, X_val, y_val)
        update_general_config(cfg_model_specs, best_threshold) # Update global models registry
    else:
        update_general_config(cfg_model_specs) # Update global models registry

if __name__ == "__main__":
    main()