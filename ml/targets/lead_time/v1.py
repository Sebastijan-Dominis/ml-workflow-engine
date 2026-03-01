from ml.targets.base import TargetStrategy
import pandas as pd

class LeadTimeTargetV1(TargetStrategy):
    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[["lead_time", "row_id"]].copy()