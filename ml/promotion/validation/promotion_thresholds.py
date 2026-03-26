"""Validation functions for promotion thresholds."""
import logging

from ml.exceptions import ConfigError
from ml.promotion.config.promotion_thresholds import PromotionThresholds

logger = logging.getLogger(__name__)

def validate_promotion_thresholds(promotion_thresholds: dict) -> PromotionThresholds:
    """Validate raw promotion thresholds payload into typed schema.

    Args:
        promotion_thresholds: Raw threshold configuration dictionary.

    Returns:
        PromotionThresholds: Validated threshold configuration object.
    """

    try:
        return PromotionThresholds(**promotion_thresholds)
    except Exception as e:
        msg = f"Invalid promotion thresholds configuration. Configuration: {promotion_thresholds}"
        logger.exception(msg)
        raise ConfigError(msg) from e
