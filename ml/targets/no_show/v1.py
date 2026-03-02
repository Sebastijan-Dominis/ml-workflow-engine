"""Version 1 no-show target strategy."""

import pandas as pd
from ml.targets.base import TargetStrategy

class NoShowTargetV1(TargetStrategy):
    """Build a binary no-show target derived from reservation status."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create no-show labels and return them with stable row identifiers.

        Args:
            data: Input booking dataframe containing reservation status and ``row_id``.

        Returns:
            Dataframe with computed ``no_show`` target labels and row identifiers.
        """

        # include row_id for tracking purposes, even though it's not part of the target variable; don't change the rest of the logic
        
        data["no_show"] = (data["reservation_status"] == "No-Show").astype(int)
        return data[["no_show", "row_id"]].copy()