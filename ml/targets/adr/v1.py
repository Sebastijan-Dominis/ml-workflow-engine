import pandas as pd

from ml.targets.base import TargetStrategy


class AdrTargetV1(TargetStrategy):
    def build(self, data: pd.DataFrame) -> pd.Series:
        return data["adr"].copy()