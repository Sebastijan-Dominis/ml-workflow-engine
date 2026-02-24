import pandas as pd
from ml.targets.base import TargetStrategy

class SpecialRequestsTargetV1(TargetStrategy):
    def build(self, data: pd.DataFrame) -> pd.Series:
        return data["total_of_special_requests"].copy()