import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any
import yaml

from ml.exceptions import ConfigError
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
