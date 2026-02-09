import logging

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

def validate_training_behavior_consistency(model_cfg: TrainModelConfig) -> None:
    if model_cfg.training.early_stopping_rounds:
        missing = []
        for fs in model_cfg.feature_store.feature_sets:
            if fs.X_val is None or fs.y_val is None:
                missing.append(fs.name)
        if missing:
            msg = f"Early stopping is enabled but the following feature sets are missing validation data: {missing}"
            logger.error(msg)
            raise ConfigError(msg)
    if model_cfg.cv <= 1 and model_cfg.training.early_stopping_rounds:
        msg = "Early stopping is enabled but cv is set to 1 or less, which is inconsistent. Please set cv > 1 for early stopping to work."
        logger.error(msg)
        raise ConfigError(msg)
    if model_cfg.seed is None:
        msg = "A random seed must be specified for training to ensure reproducibility."
        logger.error(msg)
        raise ConfigError(msg)
    
    logger.debug("Training behavior consistency validation passed.")