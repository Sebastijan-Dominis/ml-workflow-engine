"""Logical consistency checks for training behavior configuration."""

import logging

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

def validate_training_behavior_consistency(model_cfg: TrainModelConfig) -> None:
    """Validate seed and early-stopping settings are logically consistent.

    Args:
        model_cfg: Validated training model configuration.

    Returns:
        None.
    """

    if model_cfg.cv <= 1 and model_cfg.training.early_stopping_rounds:
        msg = "Early stopping is enabled but cv is set to 1 or less, which is inconsistent. Please set cv > 1 for early stopping to work."
        logger.error(msg)
        raise ConfigError(msg)
    if model_cfg.seed is None:
        msg = "A random seed must be specified for training to ensure reproducibility."
        logger.error(msg)
        raise ConfigError(msg)

    logger.debug("Training behavior consistency validation passed.")
