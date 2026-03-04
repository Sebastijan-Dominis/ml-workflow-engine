"""Helpers for resolving feature freeze strategy type from raw config."""

import logging

from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

def get_strategy_type(cfg_raw: dict) -> str:
    """Extract and validate the required strategy ``type`` field.

    Args:
        cfg_raw: Raw feature registry configuration dictionary.

    Returns:
        Strategy type string from the configuration.
    """

    if "type" not in cfg_raw:
        msg = "Missing 'type' field in feature registry config."
        logger.error(msg)
        raise ConfigError(msg)
    return cfg_raw["type"]
