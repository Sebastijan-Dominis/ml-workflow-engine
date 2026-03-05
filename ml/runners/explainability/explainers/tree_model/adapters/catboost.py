"""CatBoost-specific implementation of the tree-model adapter interface."""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from catboost import Pool
from ml.exceptions import ExplainabilityError
from ml.runners.explainability.explainers.tree_model.adapters.base import TreeModelAdapter

logger = logging.getLogger(__name__)

class CatBoostAdapter(TreeModelAdapter):
    """Adapter exposing CatBoost SHAP and feature-importance APIs."""

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for CatBoost model and return feature-only matrix.

        Args:
            X: Feature dataframe for SHAP computation.

        Returns:
            np.ndarray: SHAP value matrix excluding expected-value column.
        """
        try:
            pool = Pool(
                data=X,
                cat_features=self.model.get_cat_feature_indices()
            )
            shap_values = np.asarray(self.model.get_feature_importance(
                data=pool,
                type="ShapValues"
            ))
        except Exception as e:
            msg = "Error computing SHAP values for CatBoost model. Ensure that the input data is in the correct format and that the model supports SHAP value computation."
            logger.exception(msg)
            raise ExplainabilityError(msg) from e

        return shap_values[:, :-1]

    def compute_feature_importances(self, importance_type: Literal["PredictionValuesChange", "LossFunctionChange", "FeatureImportance", "TotalGain"] | None) -> np.ndarray:
        """Compute CatBoost feature importances for the requested type.

        Args:
            importance_type: Requested CatBoost importance type.

        Returns:
            np.ndarray: Feature importance values.
        """
        try:
            return np.asarray(self.model.get_feature_importance(type=importance_type))
        except Exception as e:
            msg = f"Error computing CatBoost feature importances with type '{importance_type}'. Ensure that the importance type is valid and supported by CatBoost."
            logger.exception(msg)
            raise ExplainabilityError(msg) from e
