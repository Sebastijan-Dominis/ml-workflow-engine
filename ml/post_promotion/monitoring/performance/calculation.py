"""Module for calculating current performance metrics for post-promotion monitoring."""
import argparse
import logging
from typing import Literal

import pandas as pd

from ml.exceptions import PipelineContractError
from ml.post_promotion.monitoring.loading.predictions import load_predictions
from ml.post_promotion.monitoring.loading.training_metrics import load_training_metrics_file
from ml.promotion.config.registry_entry import RegistryEntry
from ml.runners.evaluation.evaluators.classification.metrics import (
    compute_metrics as compute_classification_metrics,
)
from ml.runners.evaluation.evaluators.regression.metrics import (
    compute_metrics as compute_regression_metrics,
)

logger = logging.getLogger(__name__)

def calculate_current_performance(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"],
    target: pd.Series
) -> dict[str, float]:
    """Calculate current performance metrics based on predictions and target values.

    Args:
        args: Command-line arguments containing necessary identifiers.
        model_metadata: Metadata for the model being evaluated.
        stage: The stage of the model being evaluated ("production" or "staging").
        target: The target variable for the inference data.

    Returns:
        A dictionary containing the current performance metrics.
    """
    predictions = load_predictions(args, stage)
    df = predictions.join(target.rename("target"), how="inner")
    y_true = df["target"]
    y_pred = df["prediction"]
    training_metrics = load_training_metrics_file(args, model_metadata)
    if training_metrics.task_type == "classification":
        threshold = None
        threshold_info = training_metrics.metrics.get("threshold", {})
        if isinstance(threshold_info, dict):
            threshold = threshold_info.get("value")
        if threshold is None or not isinstance(threshold, (int, float)):
            msg = "Training metrics for classification task is missing 'threshold' information. Defaulting to 0.5."
            logger.critical(msg)
            threshold = 0.5
        # Consider changing for multiclass in the future
        y_prob = predictions.get("proba_1") if "proba_1" in predictions.columns else None
        current_performance = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            threshold=threshold,
        )
    elif training_metrics.task_type == "regression":
        current_performance = compute_regression_metrics(
            y_true=y_true,
            y_pred=y_pred
        )
    else:
        msg = f"Unsupported task type in training metrics: {training_metrics.task_type}"
        logger.error(msg)
        raise PipelineContractError(msg)
    return current_performance
