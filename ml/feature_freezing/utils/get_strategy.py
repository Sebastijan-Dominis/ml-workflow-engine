"""Factory helper for resolving registered feature freeze strategies."""

import logging

from ml.exceptions import UserError
from ml.feature_freezing.freeze_strategies.base import FreezeStrategy
from ml.registry.freeze_strategy_registry import STRATEGIES

logger = logging.getLogger(__name__)

def get_strategy(data_type: str) -> FreezeStrategy:
    """Instantiate strategy implementation for the requested data type.

    Args:
        data_type: Feature data type key used to resolve a freeze strategy.

    Returns:
        Instantiated freeze strategy matching ``data_type``.
    """

    strategy_cls = STRATEGIES.get(data_type)

    if not strategy_cls:
        msg = f"No freeze strategy registered for data type {data_type}."
        logger.error(msg)
        raise UserError(msg)

    strategy = strategy_cls()

    logger.debug(
        "Using freeze strategy %s for data type=%s",
        strategy.__class__.__name__,
        data_type,
    )

    return strategy