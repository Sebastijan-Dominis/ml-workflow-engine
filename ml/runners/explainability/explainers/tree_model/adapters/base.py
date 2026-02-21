from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import pandas as pd


class TreeModelAdapter(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def compute_feature_importances(self, importance_type: Optional[Literal["PredictionValuesChange", "LossFunctionChange", "FeatureImportance", "TotalGain"]]) -> np.ndarray:
        pass