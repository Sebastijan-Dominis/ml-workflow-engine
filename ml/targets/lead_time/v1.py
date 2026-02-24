from ml.targets.base import TargetStrategy
import pandas as pd

class LeadTimeTargetV1(TargetStrategy):
    def build(self, data: pd.DataFrame) -> pd.Series:
        return data["lead_time"].copy()