import logging
import hashlib

import yaml

from ml.data.utils.config.schemas.interim import InterimConfig
from ml.data.utils.config.schemas.processed import ProcessedConfig
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.exceptions import RuntimeMLException

logger = logging.getLogger(__name__)

def compute_config_hash(config: InterimConfig | ProcessedConfig | TabularFeaturesConfig) -> str:
    try:        
        config_str = yaml.dump(config, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    except Exception as e:
        msg = f"Error computing config hash. "
        logger.error(msg + f"Details: {str(e)}")
        raise RuntimeMLException(msg) from e