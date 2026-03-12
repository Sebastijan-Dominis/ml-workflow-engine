"""Validation entrypoint for model configuration dictionaries."""

import logging
from typing import Any, Literal

from pydantic_core import ValidationError

from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

def validate_model_config(cfg_raw: dict[str, Any], cfg_type: Literal["search", "train"]) -> SearchModelConfig | TrainModelConfig:
    """Validate raw model config payload with the appropriate schema.

    Args:
        cfg_raw: Raw config loaded from YAML/JSON.
        cfg_type: Type of config to validate.

    Returns:
        SearchModelConfig | TrainModelConfig: Validated config model instance.

    Raises:
        ConfigError: If cfg_type is unknown or validation fails.
    """
    cfg_raw.setdefault("_meta", {})
    try:
        validated_cfg: SearchModelConfig | TrainModelConfig
        if cfg_type == "search":
            validated_cfg = SearchModelConfig(**cfg_raw)
        elif cfg_type == "train":
            validated_cfg = TrainModelConfig(**cfg_raw)
        else:
            cfg_raw["_meta"]["validation_status"] = "failed"
            msg = f"Unknown config type: {cfg_type}"
            logger.error(msg)
            raise ConfigError(msg)

        validated_cfg.meta.validation_status = "ok"
        validated_cfg.meta.validation_errors = None
        return validated_cfg

    except ValidationError as e:
        cfg_raw["_meta"]["validation_status"] = "failed"
        logger.error("Model config validation failed for type '%s':", cfg_type)
        for err in e.errors():
            field_path = ".".join(map(str, err.get("loc", [])))
            msg = err.get("msg", "Unknown error")
            logger.error(" - Field '%s': %s", field_path, msg)

        if isinstance(cfg_raw, dict):
            cfg_raw["_meta"]["validation_errors"] = e.errors()

        raise ConfigError(f"Validation failed for {cfg_type} config") from e
