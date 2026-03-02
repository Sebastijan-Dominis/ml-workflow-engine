"""Hashing helpers for validated model configuration objects."""

import hashlib
import json
import logging
from copy import deepcopy
from typing import Any, overload

from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

def compute_config_hash(cfg: dict[str, Any]) -> str:
    """Compute SHA-256 hash for a config dictionary excluding ``_meta``.

    Args:
        cfg: Configuration dictionary payload.

    Returns:
        str: SHA-256 hash digest.
    """

    if not isinstance(cfg, dict):
        msg = "Config must be a dictionary to compute its hash."
        logger.error(msg)
        raise ConfigError(msg)

    cfg_copy = deepcopy(cfg)
    cfg_copy.pop("_meta", None)  # remove infrastructure-only metadata
    payload = json.dumps(cfg_copy, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

@overload
def add_config_hash(cfg: SearchModelConfig) -> SearchModelConfig:
    """Typed overload for applying config hash to search configuration objects.

    Args:
        cfg: Search configuration object.

    Returns:
        SearchModelConfig: Search config with hash metadata populated.
    """
    ...

@overload
def add_config_hash(cfg: TrainModelConfig) -> TrainModelConfig:
    """Typed overload for applying config hash to training configuration objects.

    Args:
        cfg: Training configuration object.

    Returns:
        TrainModelConfig: Training config with hash metadata populated.
    """
    ...

def add_config_hash(cfg: SearchModelConfig | TrainModelConfig) -> SearchModelConfig | TrainModelConfig:
    """Compute and attach config hash to ``cfg.meta.config_hash``.

    Args:
        cfg: Search or training configuration object.

    Returns:
        SearchModelConfig | TrainModelConfig: Updated config object.
    """

    runtime_dict = cfg.model_dump(exclude={"_meta"}, by_alias=True)
    config_hash = compute_config_hash(runtime_dict)

    cfg.meta.config_hash = config_hash
    return cfg