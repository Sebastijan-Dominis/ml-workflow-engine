"""Version 1 cancellation target strategy."""

import pandas as pd
from ml.targets.base import TargetStrategy


class CancellationTargetV1(TargetStrategy):
    """Build the cancellation classification target from booking records."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return cancellation labels with stable row identifiers.

        Args:
            data: Input booking dataframe containing ``is_canceled`` and ``row_id``.

        Returns:
            Dataframe with cancellation target labels and row identifiers.
        """

        return data[['is_canceled', 'row_id']].copy()
