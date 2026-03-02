"""Hash utilities for non-model Pydantic configuration objects."""

import hashlib
import logging

import yaml

from ml.data.config.schemas.interim import InterimConfig
from ml.data.config.schemas.processed import ProcessedConfig
from ml.exceptions import RuntimeMLException
from ml.feature_freezing.freeze_strategies.tabular.config.models import \
    TabularFeaturesConfig

logger = logging.getLogger(__name__)

def compute_config_hash(config: InterimConfig | ProcessedConfig | TabularFeaturesConfig) -> str:
    """Compute a deterministic MD5 hash for a typed configuration object.

    Args:
        config: Interim, processed, or tabular feature-freezing config object.

    Returns:
        str: Hex digest of the serialized configuration content.
    """
    try:        
        config_str = yaml.dump(config, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    except Exception as e:
        msg = f"Error computing config hash. "
        logger.error(msg + f"Details: {str(e)}")
        raise RuntimeMLException(msg) from e