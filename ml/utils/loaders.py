import json
import logging
from pathlib import Path
from typing import Any

import yaml

from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Config file not found: {path}"
        logger.error(msg)
        raise ConfigError(msg)

    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        msg = f"Config at {path} must be a YAML mapping"
        logger.error(msg)
        raise ConfigError(msg)

    return cfg

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Config file not found: {path}"
        logger.error(msg)
        raise ConfigError(msg)

    with path.open("r") as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in config file {path}: {e}"
            logger.error(msg)
            raise ConfigError(msg)

    if not isinstance(cfg, dict):
        msg = f"Config at {path} must be a JSON object"
        logger.error(msg)
        raise ConfigError(msg)

    return cfg