"""Validation entrypoint for feature registry strategy configs."""

import logging

from ml.exceptions import UserError
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

logger = logging.getLogger(__name__)

SCHEMAS = {
    "tabular": TabularFeaturesConfig,
}

def validate_feature_registry(raw_config: dict, data_type: str) -> TabularFeaturesConfig:
    """Validate raw feature registry payload for the given strategy type.

    Args:
        raw_config: Raw strategy config dictionary.
        data_type: Feature strategy type identifier.

    Returns:
        TabularFeaturesConfig: Validated tabular strategy configuration.
    """

    try:
        if data_type not in SCHEMAS:
            msg = f"Unsupported data type: {data_type}"
            logger.error(msg)
            raise UserError(msg)
        config = SCHEMAS[data_type](**raw_config)
        return config
    except UserError as e:
        msg = "Feature registry validation failed."
        logger.error(msg)
        raise UserError(msg) from e
