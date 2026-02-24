import pandas as pd

from ml.targets.base import TargetStrategy


class CancellationTargetV1(TargetStrategy):
    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[['is_canceled', 'row_id']].copy()