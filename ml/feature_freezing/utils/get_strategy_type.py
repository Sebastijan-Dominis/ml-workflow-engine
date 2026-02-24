import logging

from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

def get_strategy_type(cfg_raw: dict) -> str:
    if "type" not in cfg_raw:
        msg = "Missing 'type' field in feature registry config."
        logger.error(msg)
        raise ConfigError(msg)
    return cfg_raw["type"]