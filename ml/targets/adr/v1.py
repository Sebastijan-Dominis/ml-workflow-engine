"""Version 1 ADR target strategy."""

import pandas as pd

from ml.targets.base import TargetStrategy


class AdrTargetV1(TargetStrategy):
    """Build the ADR regression target from booking records."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return ADR target values with stable entity keys.

        Args:
            data: Input booking dataframe containing ``adr`` and ``entity_key``.

        Returns:
            Dataframe with ADR target values and corresponding entity keys.
        """

        return data[["adr", self.entity_key]].copy()
