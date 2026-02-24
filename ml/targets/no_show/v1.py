import pandas as pd
from ml.targets.base import TargetStrategy

class NoShowTargetV1(TargetStrategy):
    def build(self, data: pd.DataFrame) -> pd.Series:
        return (data["reservation_status"] == "No-Show").astype(int)