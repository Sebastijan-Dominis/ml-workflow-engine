"""Abstract base interface for dataset-specific row-id generators."""

from abc import ABC, abstractmethod
from typing import TypedDict

import pandas as pd


class RowIDMetadata(TypedDict):
    """Metadata about the row_id generation process for lineage tracking."""
    cols_for_row_id: list[str]
    fingerprint: str

class AddRowIDBase(ABC):
    """Base contract for row-id generation implementations."""

    @abstractmethod
    def add_row_id(self, df: pd.DataFrame) -> tuple[pd.DataFrame, RowIDMetadata]:
        """Return dataframe augmented with row identifiers and trace metadata.

        Args:
            df: Input dataframe requiring row identifier creation.

        Returns:
            Tuple of augmented dataframe and row-id lineage metadata.
        """
        pass
