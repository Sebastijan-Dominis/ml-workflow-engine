import logging
logger = logging.getLogger(__name__)

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.exceptions import UserError

def validate_feature_registry(raw_config: dict) -> TabularFeaturesConfig:
    try:
        config = TabularFeaturesConfig(**raw_config)
        return config
    except UserError as e:
        logger.error("Feature registry validation failed")
        raise UserError("Invalid feature registry configuration") from e