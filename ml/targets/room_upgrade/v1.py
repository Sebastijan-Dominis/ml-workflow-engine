import pandas as pd
from ml.targets.base import TargetStrategy

class RoomUpgradeTargetV1(TargetStrategy):
    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        data["room_upgrade"] = (data["reserved_room_type"].astype(str) != data["assigned_room_type"].astype(str)).astype(int)
        return data[["room_upgrade", "row_id"]].copy()