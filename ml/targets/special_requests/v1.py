import pandas as pd
from ml.targets.base import TargetStrategy

class SpecialRequestsTargetV1(TargetStrategy):
    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[["total_of_special_requests", "row_id"]].copy()