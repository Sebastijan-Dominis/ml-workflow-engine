"""Loading helpers for YAML/JSON configs and tabular data artifacts."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from ml.exceptions import ConfigError, DataError

FORMAT_REGISTRY_READ = {
    "parquet": pd.read_parquet,
    "csv": pd.read_csv,
    "json": pd.read_json,
    "arrow": lambda p: pd.read_feather(p),
}

logger = logging.getLogger(__name__)

def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk with validation and typed errors.

    Args:
        path: YAML file path.

    Returns:
        dict[str, Any]: Parsed YAML mapping.
    """

    if not path.exists():
        msg = f"Config file not found: {path}"
        logger.error(msg)
        raise ConfigError(msg)

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        msg = f"Config at {path} must be a YAML mapping"
        logger.error(msg)
        raise ConfigError(msg)

    return cfg

def load_json(path: Path, strict = True) -> dict[str, Any]:
    """Load a JSON object from disk, optionally returning empty dict when absent.

    Args:
        path: JSON file path.
        strict: Whether missing files should raise instead of returning `{}`.

    Returns:
        dict[str, Any]: Parsed JSON object.
    """

    if not path.exists():
        if strict:
            msg = f"File not found: {path}"
            logger.error(msg)
            raise DataError(msg) from None
        else:
            # No logger warning, since non-strict loading is used in some cases where the file may not exist on first run (e.g. loading best broad params before they are saved). Log warnings separately if needed.
            return {}

    with path.open("r", encoding="utf-8") as f:
        try:
            tgt_json = json.load(f)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in file {path}: {e}"
            logger.error(msg)
            raise ConfigError(msg) from None

    if not isinstance(tgt_json, dict):
        msg = f"Content at {path} must be a JSON object"
        logger.error(msg)
        raise ConfigError(msg) from None

    return tgt_json

def read_data(format: str, path: Path) -> pd.DataFrame:
    """Read tabular data by registered format and normalize read-time failures.

    Args:
        format: Registered file format key.
        path: Data file path.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """

    reader = FORMAT_REGISTRY_READ.get(format)
    if not reader:
        msg = f"Unsupported data format: {format}"
        logger.error(msg)
        raise ConfigError(msg) from None

    try:
        if format == "csv":
            df = reader(path, na_values=['', 'NA', 'nan'], keep_default_na=True)
        else:
            df = reader(path)
        logger.debug(f"Successfully read data in format '{format}' from {path}.")
        return df
    except Exception as e:
        msg = f"Error reading data in format '{format}' from {path}."
        logger.exception(msg)
        raise DataError(msg) from e
