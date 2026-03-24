"""Version 1 no-show target strategy."""

import pandas as pd

from ml.targets.base import TargetStrategy


class NoShowTargetV1(TargetStrategy):
    """Build a binary no-show target derived from reservation status."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create no-show labels and return them with stable entity keys.

        Args:
            data: Input booking dataframe containing reservation status and ``entity_key``.

        Returns:
            Dataframe with computed ``no_show`` target labels and entity keys.
        """

        data["no_show"] = (data["reservation_status"] == "No-Show").astype(int)
        return data[["no_show", self.entity_key]].copy()
