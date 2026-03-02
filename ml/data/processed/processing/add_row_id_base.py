"""Abstract base interface for dataset-specific row-id generators."""

import pandas as pd


class AddRowIDBase():
    """Base contract for row-id generation implementations."""

    def add_row_id(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Return dataframe augmented with row identifiers and trace metadata.

        Args:
            df: Input dataframe requiring row identifier creation.

        Returns:
            Tuple of augmented dataframe and row-id lineage metadata.
        """
        ...