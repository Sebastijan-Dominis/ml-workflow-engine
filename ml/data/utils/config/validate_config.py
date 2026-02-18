import logging
from typing import Literal, overload

from ml.data.utils.config.schemas.interim import InterimConfig
from ml.data.utils.config.schemas.processed import ProcessedConfig
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

@overload
def validate_config(config: dict, type: Literal["interim"]) -> InterimConfig: ...

@overload
def validate_config(config: dict, type: Literal["processed"]) -> ProcessedConfig: ...

def validate_config(config: dict, type: Literal["interim", "processed"]) -> InterimConfig | ProcessedConfig:
    try:
        if type == "interim":
            return InterimConfig(**config)
        elif type == "processed":
            return ProcessedConfig(**config)
        else:
            msg = f"Unsupported config type '{type}'. Supported types are 'interim' and 'processed'."
            logger.error(msg)
            raise ConfigError(msg)
    except Exception as e:
        msg = f"Configuration validation error. "
        logger.error(msg + f"Details: {str(e)}")
        raise ConfigError(msg) from e