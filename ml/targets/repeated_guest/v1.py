import pandas as pd
from ml.targets.base import TargetStrategy

class RepeatedGuestTargetV1(TargetStrategy):
    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[["is_repeated_guest", "row_id"]].copy()