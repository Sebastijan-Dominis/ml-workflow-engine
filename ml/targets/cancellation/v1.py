import pandas as pd

from ml.targets.base import TargetStrategy


class CancellationTargetV1(TargetStrategy):
    def build(self, data: pd.DataFrame) -> pd.Series:
        return data['is_canceled'].copy()