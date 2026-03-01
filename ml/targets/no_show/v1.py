import pandas as pd
from ml.targets.base import TargetStrategy

class NoShowTargetV1(TargetStrategy):
    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        # include row_id for tracking purposes, even though it's not part of the target variable; don't change the rest of the logic
        
        data["no_show"] = (data["reservation_status"] == "No-Show").astype(int)
        return data[["no_show", "row_id"]].copy()