"""This module contains the function to validate the incoming configuration payload for promotion thresholds."""
from ml.promotion.config.models import PromotionThresholds


def validate_config_payload(payload: dict) -> PromotionThresholds:
    """Validate the incoming payload against the PromotionThresholds schema.

    Args:
        payload (dict): The incoming configuration payload to validate.

    Returns:
        PromotionThresholds: An instance of the PromotionThresholds model if validation is successful.
    """
    return PromotionThresholds(**payload)
