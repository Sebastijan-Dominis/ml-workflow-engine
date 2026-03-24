"""Base abstractions for target construction strategies."""

import logging
from abc import ABC, abstractmethod

import pandas as pd

from ml.exceptions import UserError

logger = logging.getLogger(__name__)

class TargetStrategy(ABC):
    """Abstract strategy for building model targets from prepared data."""
    def __init__(self, entity_key: str) -> None:
        """Initialize the target strategy with a specified entity key column name.

        Args:
            entity_key: The name of the entity key column to use for traceability.
        """
        self.entity_key = entity_key
        self.REQUIRED_COLUMNS = {self.entity_key}

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input frame and delegate target construction to implementation.

        Args:
            data: Source dataframe containing target columns.
            entity_key: The name of the entity key column to extract.

        Returns:
            pd.DataFrame: Built target dataframe with `entity_key`.
        """

        self._validate(data)
        return self._build(data)

    def _validate(self, data: pd.DataFrame) -> None:
        """Ensure shared required columns are present before target extraction.

        Args:
            data: Source dataframe.

        Returns:
            None: Raises when required columns are missing.
        """

        missing = self.REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise UserError(
                f"Target data missing required columns: {missing}"
            )

    @abstractmethod
    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build and return a target dataframe with `entity_key` for traceability.

        Args:
            data: Source dataframe containing target inputs.

        Returns:
            pd.DataFrame: Target dataframe including `entity_key`.
        """

        pass
