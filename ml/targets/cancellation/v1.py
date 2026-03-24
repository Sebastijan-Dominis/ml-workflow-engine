"""Version 1 cancellation target strategy."""

import pandas as pd

from ml.targets.base import TargetStrategy


class CancellationTargetV1(TargetStrategy):
    """Build the cancellation classification target from booking records."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return cancellation labels with stable entity keys.

        Args:
            data: Input booking dataframe containing ``is_canceled`` and ``entity_key``.

        Returns:
            Dataframe with cancellation target labels and entity keys.
        """

        return data[['is_canceled', self.entity_key]].copy()
