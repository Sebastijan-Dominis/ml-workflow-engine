"""Metric computation and split evaluation helpers for regression tasks."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from ml.config.schemas.model_specs import TargetTransformConfig
from ml.exceptions import EvaluationError
from ml.features.transforms.transform_target import inverse_transform_target
from ml.runners.evaluation.constants.data_splits import DataSplits
from ml.runners.evaluation.models.predictions import PredictionArtifacts
from ml.runners.evaluation.utils.get_entity_keys import get_entity_keys
from ml.runners.shared.formatting.ensure_1d_array import ensure_1d_array

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Compute core regression metrics and residual summary statistics.

    Args:
        y_true: Ground-truth regression targets.
        y_pred: Predicted regression values.

    Returns:
        dict[str, float]: Computed regression metrics.
    """

    metrics: dict[str, float] = {}

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["mse"] = float(mse)
    metrics["rmse"] = float(rmse)
    metrics["r2"] = float(r2_score(y_true, y_pred))
    metrics["median_ae"] = float(median_absolute_error(y_true, y_pred))
    metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred))

    # Optional but very useful
    residuals = y_true - y_pred
    metrics["residual_mean"] = float(residuals.mean())
    metrics["residual_std"] = float(residuals.std())

    return metrics


def evaluate_split(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    split_entity_keys: pd.Series,
    split_name: str,
    transform_cfg: TargetTransformConfig,
    entity_key: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate one split and return metrics and per-row prediction output.

    Args:
        pipeline: Fitted model pipeline.
        X: Split features.
        y: Split target values.
        split_entity_keys: Row identifiers for split records.
        split_name: Split label.
        transform_cfg: Target-transform configuration for inverse transform.
        entity_key: The name of the entity key column to extract.

    Returns:
        tuple[dict[str, float], pd.DataFrame]: Split metrics and prediction dataframe.
    """

    # Generate predictions
    y_pred = pipeline.predict(X)

    y_pred = ensure_1d_array(y_pred)

    y_pred = inverse_transform_target(
        y_pred,
        transform_config=transform_cfg,
        split_name=split_name
    )

    y_pred = pd.Series(y_pred, index=y.index, name="y_pred")

    # Compute metrics
    logger.info(f"Computing regression metrics for the {split_name} split...")
    metrics = compute_metrics(y, y_pred)

    # Create prediction dataframe
    df_preds = pd.DataFrame({
        "entity_key": split_entity_keys,
        "split": split_name,
        "y_true": y,
        "y_pred": y_pred,
        "residual": y - y_pred,
    })

    return metrics, df_preds


def evaluate_model(
    *,
    pipeline: Pipeline,
    data_splits: DataSplits,
    transform_cfg: TargetTransformConfig,
    entity_key: str,
) -> tuple[dict[str, dict[str, float]], PredictionArtifacts]:
    """Evaluate all splits and aggregate regression metrics/predictions.

    Args:
        pipeline: Fitted model pipeline.
        data_splits: Dataset split container.
        transform_cfg: Target-transform configuration for inverse transform.

    Returns:
        tuple[dict[str, dict[str, float]], PredictionArtifacts]: Metrics and predictions by split.
    """

    evaluation_metrics: dict[str, dict[str, float]] = {}
    prediction_dfs_raw: dict[str, pd.DataFrame] = {}

    for split_name, (X, y) in data_splits.__dict__.items():

        split_entity_keys = get_entity_keys(X, entity_key=entity_key)

        if entity_key in X.columns:
            X = X.drop(columns=[entity_key])

        logger.debug(
            f"Evaluating regression split '{split_name}' with {len(y)} samples."
        )

        metrics, df_preds = evaluate_split(
            pipeline=pipeline,
            X=X,
            y=y,
            split_entity_keys=split_entity_keys,
            split_name=split_name,
            transform_cfg=transform_cfg,
            entity_key=entity_key
        )

        evaluation_metrics[split_name] = metrics
        prediction_dfs_raw[split_name] = df_preds

    try:
        prediction_dfs = PredictionArtifacts(**prediction_dfs_raw)
    except Exception as e:
        msg = f"Error constructing PredictionArtifacts with prediction dataframes: {prediction_dfs_raw}."
        logger.exception(msg)
        raise EvaluationError(msg) from e

    return evaluation_metrics, prediction_dfs
