"""Validation entrypoint for interim and processed data configs."""

import logging
from typing import Literal, overload

from ml.data.config.schemas.interim import InterimConfig
from ml.data.config.schemas.processed import ProcessedConfig
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

@overload
def validate_config(config: dict, type: Literal["interim"]) -> InterimConfig:
    """Typed overload for validating interim data configuration payloads.

    Args:
        config: Raw configuration payload.
        type: Config type discriminator.

    Returns:
        InterimConfig: Validated interim configuration object.
    """
    ...

@overload
def validate_config(config: dict, type: Literal["processed"]) -> ProcessedConfig:
    """Typed overload for validating processed data configuration payloads.

    Args:
        config: Raw configuration payload.
        type: Config type discriminator.

    Returns:
        ProcessedConfig: Validated processed configuration object.
    """
    ...

def validate_config(config: dict, type: Literal["interim", "processed"]) -> InterimConfig | ProcessedConfig:
    """Validate raw config dictionary using stage-specific Pydantic schema.

    Args:
        config: Raw configuration payload.
        type: Config type selector (``"interim"`` or ``"processed"``).

    Returns:
        InterimConfig | ProcessedConfig: Typed validated configuration object.
    """
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
        msg = "Configuration validation error. "
        logger.error(msg + f"Details: {str(e)}")
        raise ConfigError(msg) from e
