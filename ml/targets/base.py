import logging
from abc import ABC, abstractmethod

import pandas as pd

from ml.exceptions import UserError

logger = logging.getLogger(__name__)

class TargetStrategy(ABC):
    REQUIRED_COLUMNS = {'row_id'}

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        return self._build(data)

    def _validate(self, data: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise UserError(
                f"Target data missing required columns: {missing}"
            )

    @abstractmethod
    def _build(self, data: pd.DataFrame) -> pd.DataFrame:
        pass