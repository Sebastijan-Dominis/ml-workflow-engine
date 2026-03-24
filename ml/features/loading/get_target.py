"""Utilities for resolving and building targets from registered strategies."""

import logging

import pandas as pd

from ml.exceptions import ConfigError
from ml.targets.adr.v1 import AdrTargetV1
from ml.targets.base import TargetStrategy
from ml.targets.cancellation.v1 import CancellationTargetV1
from ml.targets.lead_time.v1 import LeadTimeTargetV1
from ml.targets.no_show.v1 import NoShowTargetV1
from ml.targets.repeated_guest.v1 import RepeatedGuestTargetV1
from ml.targets.room_upgrade.v1 import RoomUpgradeTargetV1
from ml.targets.special_requests.v1 import SpecialRequestsTargetV1

TARGET_STRATEGIES: dict[
    tuple[str, str],
    type[TargetStrategy]
] = {
    ("adr", "v1"): AdrTargetV1,
    ("lead_time", "v1"): LeadTimeTargetV1,
    ("no_show", "v1"): NoShowTargetV1,
    ("is_repeated_guest", "v1"): RepeatedGuestTargetV1,
    ("room_upgrade", "v1"): RoomUpgradeTargetV1,
    ("total_of_special_requests", "v1"): SpecialRequestsTargetV1,
    ("is_canceled", "v1"): CancellationTargetV1,
}

logger = logging.getLogger(__name__)

def get_target_with_entity_key(data: pd.DataFrame, key: tuple[str, str], entity_key: str) -> pd.DataFrame:
    """Build target dataframe with `entity_key` using a registry-resolved strategy.

    Args:
        data: Source dataframe containing raw columns required for target derivation.
        key: Target strategy registry key as ``(target_name, target_version)``.
        entity_key: The name of the entity key column to use for traceability.

    Returns:
        pd.DataFrame: Target dataframe including the `entity_key` column.
    """

    if key not in TARGET_STRATEGIES:
        msg = f"Target strategy for key {key} not found in registry."
        logger.error(msg)
        raise ConfigError(msg)

    target_strategy_cls = TARGET_STRATEGIES[key]
    target_strategy = target_strategy_cls(entity_key=entity_key)
    return target_strategy.build(data)
