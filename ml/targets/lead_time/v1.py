"""Version 1 lead-time target strategy."""

from ml.targets.base import TargetStrategy
import pandas as pd

class LeadTimeTargetV1(TargetStrategy):
    """Build the lead-time regression target from booking records."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return lead-time values with stable row identifiers.

        Args:
            data: Input booking dataframe containing ``lead_time`` and ``row_id``.

        Returns:
            Dataframe with lead-time target values and row identifiers.
        """

        return data[["lead_time", "row_id"]].copy()