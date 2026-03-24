"""Version 1 repeated-guest target strategy."""

import pandas as pd

from ml.targets.base import TargetStrategy


class RepeatedGuestTargetV1(TargetStrategy):
    """Build the repeated-guest classification target from booking records."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return repeated-guest labels with stable entity keys.

        Args:
            data: Input booking dataframe containing ``is_repeated_guest`` and ``entity_key``.

        Returns:
            Dataframe with repeated-guest target labels and entity keys.
        """

        return data[["is_repeated_guest", self.entity_key]].copy()
