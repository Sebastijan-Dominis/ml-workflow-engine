"""Base abstractions for target construction strategies."""

import logging
from abc import ABC, abstractmethod

import pandas as pd
from ml.exceptions import UserError

logger = logging.getLogger(__name__)

class TargetStrategy(ABC):
    """Abstract strategy for building model targets from prepared data."""

    REQUIRED_COLUMNS = {'row_id'}

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input frame and delegate target construction to implementation.

        Args:
            data: Source dataframe containing target columns.

        Returns:
            pd.DataFrame: Built target dataframe with `row_id`.
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
        """Build and return a target dataframe with `row_id` for traceability.

        Args:
            data: Source dataframe containing target inputs.

        Returns:
            pd.DataFrame: Target dataframe including `row_id`.
        """

        pass
