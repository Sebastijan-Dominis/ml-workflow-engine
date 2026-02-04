import logging
logger = logging.getLogger(__name__)
from typing import Dict, Literal, Any
from pydantic_core import ValidationError

from ml.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import ConfigError

def validate_model_config(cfg_raw: Dict[str, Any], cfg_type: Literal["search", "train"]) -> Dict[str, Any]:
    """
    Validate a raw model config dict using the appropriate Pydantic schema.

    Args:
        cfg_raw (Dict[str, Any]): Raw config loaded from YAML/JSON.
        cfg_type (Literal["search", "train"]): Type of config to validate.

    Returns:
        Dict[str, Any]: Validated config as a dictionary.

    Raises:
        ConfigError: If cfg_type is unknown or validation fails.
    """
    cfg_raw.setdefault("_meta", {})
    try:
        if cfg_type == "search":
            validated_cfg = SearchModelConfig(**cfg_raw)
        elif cfg_type == "train":
            validated_cfg = TrainModelConfig(**cfg_raw)
        else:
            cfg_raw["_meta"]["validation_status"] = "failed"
            msg = f"Unknown config type: {cfg_type}"
            logger.error(msg)
            raise ConfigError(msg)

        cfg_raw["_meta"]["validation_status"] = "ok"
        cfg_raw["_meta"].pop("validation_errors", None)  # Clear previous errors if any
        return validated_cfg.model_dump()

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