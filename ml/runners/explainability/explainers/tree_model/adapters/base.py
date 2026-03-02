"""Abstract adapter interface for tree-model explainability backends."""

from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np
import pandas as pd


class TreeModelAdapter(ABC):
    """Adapter contract exposing SHAP and feature-importance operations."""

    def __init__(self, model):
        """Store wrapped model instance for downstream explainability calls.

        Args:
            model: Wrapped tree-model instance.

        Returns:
            None: Initializes adapter state.
        """

        self.model = model

    @abstractmethod
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP value matrix for provided feature dataframe.

        Args:
            X: Feature dataframe for SHAP computation.

        Returns:
            np.ndarray: SHAP value matrix.
        """
        pass

    @abstractmethod
    def compute_feature_importances(self, importance_type: Optional[Literal["PredictionValuesChange", "LossFunctionChange", "FeatureImportance", "TotalGain"]]) -> np.ndarray:
        """Compute model feature-importance vector for requested importance type.

        Args:
            importance_type: Importance definition requested by backend.

        Returns:
            np.ndarray: Feature-importance vector.
        """
        pass