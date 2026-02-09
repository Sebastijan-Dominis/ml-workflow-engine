import logging

from ml.exceptions import UserError
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

logger = logging.getLogger(__name__)

SCHEMAS = {
    "tabular": TabularFeaturesConfig,
}

def validate_feature_registry(raw_config: dict, data_type: str) -> TabularFeaturesConfig:
    try:
        if data_type not in SCHEMAS:
            msg = f"Unsupported data type: {data_type}"
            logger.error(msg)
            raise UserError(msg)
        config = SCHEMAS[data_type](**raw_config)
        return config
    except UserError as e:
        msg = f"Feature registry validation failed."
        logger.error(msg)
        raise UserError(msg) from e