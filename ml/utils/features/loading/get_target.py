"""Utilities for resolving and building targets from registered strategies."""

import logging

import pandas as pd

from ml.exceptions import ConfigError
from ml.registry.target_strategies import TARGET_STRATEGIES

logger = logging.getLogger(__name__)

def get_target_with_row_id(data: pd.DataFrame, key: tuple[str, str]) -> pd.DataFrame:
    """Build target dataframe with `row_id` using a registry-resolved strategy.

    Args:
        data: Source dataframe containing raw columns required for target derivation.
        key: Target strategy registry key as ``(target_name, target_version)``.

    Returns:
        pd.DataFrame: Target dataframe including the `row_id` column.
    """

    if key not in TARGET_STRATEGIES:
        msg = f"Target strategy for key {key} not found in registry."
        logger.error(msg)
        raise ConfigError(msg)
    
    target_strategy_cls = TARGET_STRATEGIES[key]
    target_strategy = target_strategy_cls()
    return target_strategy.build(data)