"""Module for comparing feature distributions between training and inference data in post-promotion monitoring."""
import logging
from typing import Literal

import pandas as pd

from ml.post_promotion.monitoring.feature_drifting.computations import compute_drift

logger = logging.getLogger(__name__)

def compare_feature_distributions(
    *,
    stage: Literal["production", "staging"],
    inference_features: pd.DataFrame,
    training_features: pd.DataFrame
):
    """Compare feature distributions between training and inference data.

    Args:
        stage: The stage for which to compare feature distributions ('production' or 'staging').
        inference_features: DataFrame containing the inference features.
        training_features: DataFrame containing the training features.

    Returns:
        A dictionary mapping feature names to their computed drift scores.
    """
    logger.info(f"Comparing feature distributions for the {stage} model...")

    drift_results = {}
    for col in training_features.columns:
        expected = training_features[col]
        actual = inference_features[col]
        drift_score = compute_drift(expected, actual)
        drift_results[col] = drift_score

    return drift_results
