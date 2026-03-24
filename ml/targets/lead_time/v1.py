"""Version 1 lead-time target strategy."""

import pandas as pd

from ml.targets.base import TargetStrategy


class LeadTimeTargetV1(TargetStrategy):
    """Build the lead-time regression target from booking records."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return lead-time values with stable entity keys.

        Args:
            data: Input booking dataframe containing ``lead_time`` and ``entity_key``.

        Returns:
            Dataframe with lead-time target values and entity keys.
        """

        return data[["lead_time", self.entity_key]].copy()
