"""Version 1 repeated-guest target strategy."""

import pandas as pd

from ml.targets.base import TargetStrategy


class RepeatedGuestTargetV1(TargetStrategy):
    """Build the repeated-guest classification target from booking records."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return repeated-guest labels with stable row identifiers.

        Args:
            data: Input booking dataframe containing ``is_repeated_guest`` and ``row_id``.

        Returns:
            Dataframe with repeated-guest target labels and row identifiers.
        """

        return data[["is_repeated_guest", "row_id"]].copy()
