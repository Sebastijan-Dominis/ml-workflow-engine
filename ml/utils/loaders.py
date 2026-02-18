import json
import logging
from pathlib import Path
from typing import Any

import yaml
import pandas as pd

from ml.exceptions import ConfigError
from ml.registry.format_registry import FORMAT_REGISTRY_READ

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

def read_data(format: str, path: Path) -> pd.DataFrame:
    reader = FORMAT_REGISTRY_READ.get(format)
    if not reader:
        msg = f"Unsupported data format: {format}"
        logger.error(msg)
        raise ValueError(msg)
    
    try:
        df = reader(path)
        logger.info(f"Successfully read data in format '{format}' from {path}.")
        return df
    except Exception as e:
        msg = f"Error reading data in format '{format}' from {path}. Details: {str(e)}"
        logger.error(msg)
        raise IOError(msg) from e