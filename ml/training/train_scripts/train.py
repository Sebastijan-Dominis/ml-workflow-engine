"""Training script entry point and helpers.

This module provides a small CLI entrypoint to run model training for
different tasks and algorithms. It exposes helper functions to parse
command-line arguments, load YAML configuration files, validate them
against the project's Pydantic schema, and run the appropriate trainer
implementation. After training, the resulting pipeline and metadata are
persisted and the global models registry is updated.

Typical usage:
    python -m ml.training.train_scripts.train --name_and_version cancellation_v1

The script expects a YAML config at ``ml/training/train_configs/{name_and_version}.yaml``.
"""

# General imports
import logging
logger = logging.getLogger(__name__)
import sys
import yaml
import argparse

from pydantic_core import ValidationError

# Specific training script imports
from ml.training.train_scripts.custom_training_scripts.binary_classification_catboost import (
    train_binary_classification_with_catboost
)

from ml.training.train_scripts.schemas.binary_classification_catboost_schemas import ConfigSchema

# Persistence imports
from ml.training.train_scripts.persistence.save_pipeline_and_metadata import (
    save_pipeline_and_metadata
)

from ml.training.train_scripts.persistence.update_general_config import (
    update_general_config
)

# Logger import
from ml.logging_config import setup_logging

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with attribute ``name_and_version``
            containing the model name and version in the form ``name_version``
            (for example, ``cancellation_v1``).
    """

    parser = argparse.ArgumentParser(description="Train a model.")

    parser.add_argument(
        "--name_and_version",
        type=str,
        required=True,
        help="Model name and version in the format 'name_version', e.g., 'cancellation_v1'"
    )

    return parser.parse_args()

def load_config(name_and_version: str) -> dict:   
    """Load a YAML training configuration file.

    Args:
        name_and_version (str): Model identifier with version (``name_version``).

    Returns:
        dict: Raw configuration loaded from the YAML file.

    Raises:
        FileNotFoundError: If the expected config file does not exist.
        yaml.YAMLError: If the YAML content cannot be parsed.
    """

    with open(f"ml/training/train_configs/{name_and_version}.yaml") as f:
        cfg = yaml.safe_load(f)

    return cfg

def validate_config_schema(cfg_raw: dict) -> dict:
    """Validate a raw configuration dict against the Pydantic schema.

    Args:
        cfg_raw (dict): The unvalidated configuration dictionary.

    Returns:
        dict: The validated configuration as a standard dictionary.

    Side effects:
        Logs validation errors and exits the process with status code 1 if
        the provided configuration is invalid.
    """

    try:
        cfg = ConfigSchema(**cfg_raw).model_dump()
        return cfg
    except ValidationError as e:
        logger.error("Config validation failed:")
        for err in e.errors():
            logger.error("Field %s: %s", ".".join(map(str, err['loc'])), err['msg'])
        sys.exit(1)  # Stop execution if config is invalid

def main() -> None:
    """Entrypoint to run model training according to a YAML configuration.

    The function performs the following high-level steps:
    1. Set up logging.
    2. Parse CLI arguments to obtain the config file name.
    3. Load and validate the YAML configuration.
    4. Select the appropriate trainer implementation based on the
       ``task`` and ``model.algorithm`` specified in the config.
    5. Execute training, persist the resulting pipeline and metadata,
       and update the global models registry.

    Raises:
        ValueError: If no trainer is registered for the requested task/algorithm.
    """

    setup_logging() # Set up logging configuration

    args = parse_args() # Parse command-line arguments

    cfg_raw = load_config(args.name_and_version) # Load raw YAML config

    cfg = validate_config_schema(cfg_raw) # Validate config schema

    task = cfg["task"]
    algorithm = cfg["model"]["algorithm"]

    # Trainer registry: extend this dict when adding new tasks/algorithms
    TRAINERS = {
        "binary_classification_catboost": train_binary_classification_with_catboost
    }

    key = f"{task}_{algorithm}"
    trainer = TRAINERS.get(key)

    if trainer:
        pipeline = trainer(args.name_and_version, cfg)
    else:
        raise ValueError(f"Unsupported task and algorithm: {task}_{algorithm}")
    
    save_pipeline_and_metadata(pipeline, cfg) # Persist pipeline and metadata

    update_general_config(cfg) # Update global models registry

if __name__ == "__main__":
    main()