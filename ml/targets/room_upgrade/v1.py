"""Version 1 room-upgrade target strategy."""

import pandas as pd

from ml.targets.base import TargetStrategy


class RoomUpgradeTargetV1(TargetStrategy):
    """Build a binary room-upgrade target by comparing reserved and assigned rooms."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create room-upgrade labels and return them with stable entity keys.

        Args:
            data: Input booking dataframe containing reserved/assigned room columns and ``entity_key``.

        Returns:
            Dataframe with computed ``room_upgrade`` labels and entity keys.
        """

        data["room_upgrade"] = (data["reserved_room_type"].astype(str) != data["assigned_room_type"].astype(str)).astype(int)
        return data[["room_upgrade", self.entity_key]].copy()
