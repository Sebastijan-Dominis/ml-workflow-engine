import pandas as pd
from ml.targets.base import TargetStrategy

class RoomUpgradeTargetV1(TargetStrategy):
    def build(self, data: pd.DataFrame) -> pd.Series:
        return (
            data["reserved_room_type"].astype(str)
            != data["assigned_room_type"].astype(str)
        ).astype(int)