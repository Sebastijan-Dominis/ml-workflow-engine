"""Orchestrator for logical consistency validations on training config."""

from pathlib import Path

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.runners.training.utils.logical_config_checks.validations.validate_allowed_params import (
    validate_allowed_params,
)
from ml.runners.training.utils.logical_config_checks.validations.validate_training_behavior_consistency import (
    validate_training_behavior_consistency,
)


def validate_logical_config(model_cfg: TrainModelConfig, search_dir: Path) -> None:
    """Perform logical consistency checks on the training configuration.

    This function runs a series of validations to ensure that the provided
    training configuration is logically consistent and adheres to expected
    constraints. It checks for allowed parameters, consistency of training
    behavior settings, and any other logical rules defined for the training
    process.

    Args:
        model_cfg (TrainModelConfig): The validated training configuration object.
        search_dir (Path): The directory where search artifacts are stored, used for lineage checks.

    Raises:
        ConfigError: If any logical inconsistency is detected in the configuration.
    """
    validate_allowed_params(model_cfg, search_dir)
    validate_training_behavior_consistency(model_cfg)
