import logging

import pandas as pd

from ml.exceptions import ConfigError
from ml.registry.target_strategies import TARGET_STRATEGIES

logger = logging.getLogger(__name__)

def get_target(data: pd.DataFrame, key: tuple[str, str]) -> pd.Series:
    if key not in TARGET_STRATEGIES:
        msg = f"Target strategy for key {key} not found in registry."
        logger.error(msg)
        raise ConfigError(msg)
    
    target_strategy_cls = TARGET_STRATEGIES[key]
    target_strategy = target_strategy_cls()
    return target_strategy.build(data)