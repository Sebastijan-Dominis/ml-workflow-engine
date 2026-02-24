from abc import ABC, abstractmethod

import pandas as pd


class TargetStrategy(ABC):
    @abstractmethod
    def build(self, data: pd.DataFrame) -> pd.Series:
        pass