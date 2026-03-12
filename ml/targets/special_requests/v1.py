"""Version 1 special-requests target strategy."""

import pandas as pd

from ml.targets.base import TargetStrategy


class SpecialRequestsTargetV1(TargetStrategy):
    """Build the special-requests count target from booking records."""

    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return special-request counts with stable row identifiers.

        Args:
            data: Input booking dataframe containing ``total_of_special_requests`` and ``row_id``.

        Returns:
            Dataframe with special-request target values and row identifiers.
        """

        return data[["total_of_special_requests", "row_id"]].copy()
