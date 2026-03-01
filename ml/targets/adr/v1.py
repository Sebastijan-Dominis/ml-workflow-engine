import pandas as pd

from ml.targets.base import TargetStrategy


class AdrTargetV1(TargetStrategy):
    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[["adr", "row_id"]].copy()