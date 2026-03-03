"""SHAP-based importance calculation helpers for tree-model explainability."""

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import ConfigError, DataError, ExplainabilityError
from ml.runners.explainability.explainers.tree_model.adapters.base import \
    TreeModelAdapter
from ml.runners.explainability.explainers.tree_model.utils.validators.validate_lengths import \
    validate_lengths

logger = logging.getLogger(__name__)

def get_shap_importances(
    *, 
    feature_names: NDArray[np.str_], 
    model_configs: TrainModelConfig, 
    top_k: int, 
    X_test_transformed: pd.DataFrame,
    adapter: TreeModelAdapter
    ) -> pd.DataFrame | None:
    """Compute top-k mean absolute SHAP importances for transformed test data.

    Args:
        feature_names: Transformed feature-name array.
        model_configs: Validated training model configuration.
        top_k: Number of top features to return.
        X_test_transformed: Transformed test feature dataframe.
        adapter: Tree-model adapter implementing SHAP computation.

    Returns:
        Top-k SHAP-importance dataframe, or ``None`` when disabled.
    """

    if not model_configs.explainability.methods.shap.enabled:
        logger.warning("SHAP method is not enabled in the configuration. Skipping SHAP importance computation.")
        return None

    shap_method = model_configs.explainability.methods.shap.approximate
    logger.info(f"Calculating SHAP importances using shap method: '{shap_method}'...")

    n = min(1000, X_test_transformed.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_test_transformed.shape[0], size=n, replace=False)

    if not hasattr(X_test_transformed, "iloc"):
        msg = "Transformed test data is not a pandas DataFrame. SHAP analysis requires the transformed data to be a DataFrame with column names."
        logger.error(msg)
        raise DataError(msg)
    
    X_test_sample = X_test_transformed.iloc[idx]
    
    if shap_method == "tree":
        try:
            shap_values = adapter.compute_shap_values(X_test_sample)
        except Exception:
            msg = "Error calculating SHAP values using model adapter. Ensure that the model is compatible with Tree SHAP and that the input data is correctly preprocessed."
            logger.exception(msg)
            raise ExplainabilityError(msg)

    else:
        msg = f"Unsupported SHAP method: {shap_method}. Currently, only 'tree' is supported for CatBoost models."
        logger.error(msg)
        raise ConfigError(msg)

    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_values = np.abs(shap_values)

    importances = shap_values.mean(axis=0)

    validate_lengths(feature_names, importances)

    top_k_shap_importances = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': importances,
    }).sort_values(by='mean_abs_shap', ascending=False).head(top_k)

    return top_k_shap_importances