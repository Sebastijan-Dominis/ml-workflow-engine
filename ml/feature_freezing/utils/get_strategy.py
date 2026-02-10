import logging

from ml.registry.freeze_strategy_registry import STRATEGIES
from ml.exceptions import UserError
from ml.feature_freezing.freeze_strategies.base import FreezeStrategy

logger = logging.getLogger(__name__)

def get_strategy(data_type: str) -> FreezeStrategy:
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