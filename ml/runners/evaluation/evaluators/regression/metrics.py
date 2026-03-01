import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, median_absolute_error,
                             r2_score)
from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_specs import TargetTransformConfig
from ml.runners.evaluation.constants.data_splits import DataSplits
from ml.runners.evaluation.utils.get_row_ids import get_row_ids
from ml.utils.experiments.ensure_1d_array import ensure_1d_array
from ml.utils.features.transform_target import inverse_transform_target

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:

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
    split_row_ids: pd.Series,
    split_name: str,
    transform_cfg: TargetTransformConfig,
) -> tuple[dict[str, float], pd.DataFrame]:

    # Generate predictions
    y_pred = pipeline.predict(X)

    y_pred = ensure_1d_array(y_pred)

    y_pred = inverse_transform_target(y_pred, transform_cfg)

    y_pred = pd.Series(y_pred, index=y.index, name="y_pred")

    # Compute metrics
    metrics = compute_metrics(y, y_pred)

    # Create prediction dataframe
    df_preds = pd.DataFrame({
        "row_id": split_row_ids,
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
) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:

    evaluation_metrics: dict[str, dict[str, float]] = {}
    prediction_dfs: dict[str, pd.DataFrame] = {}

    for split_name, (X, y) in data_splits.__dict__.items():

        split_row_ids = get_row_ids(X)

        if "row_id" in X.columns:
            X = X.drop(columns=["row_id"])

        logger.debug(
            f"Evaluating regression split '{split_name}' with {len(y)} samples."
        )

        metrics, df_preds = evaluate_split(
            pipeline=pipeline,
            X=X,
            y=y,
            split_row_ids=split_row_ids,
            split_name=split_name,
            transform_cfg=transform_cfg,
        )

        evaluation_metrics[split_name] = metrics
        prediction_dfs[split_name] = df_preds

    return evaluation_metrics, prediction_dfs