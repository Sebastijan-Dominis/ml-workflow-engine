import logging
logger = logging.getLogger(__name__)
import json
import hashlib
from typing import Any
from copy import deepcopy

from ml.exceptions import ConfigError

def compute_config_hash(cfg: dict[str, Any]) -> str:
    if not isinstance(cfg, dict):
        msg = "Config must be a dictionary to compute its hash."
        logger.error(msg)
        raise ConfigError(msg)

    cfg_copy = deepcopy(cfg)
    cfg_copy.pop("_meta", None)  # remove infrastructure-only metadata
    payload = json.dumps(cfg_copy, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()