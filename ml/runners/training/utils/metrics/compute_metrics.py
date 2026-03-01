import logging
from typing import Any

import pandas as pd
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             root_mean_squared_error)
from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import UserError
from ml.registry.tasks_supporting_thresholds import TASKS_SUPPORTING_THRESHOLDS
from ml.runners.training.utils.metrics.best_f1 import get_best_f1_thresh
from ml.utils.experiments.ensure_1d_array import ensure_1d_array
from ml.utils.features.transform_target import inverse_transform_target


def compute_metrics(
    *, 
    model: Any, 
    pipeline: Pipeline, 
    model_cfg: TrainModelConfig, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series
) -> dict[str, float]:
    if model_cfg.task.type == "classification":
        best_iter = model.get_best_iteration()
        train_pred = pipeline.predict_proba(X_train, ntree_end=best_iter)[:, 1]
        val_pred = pipeline.predict_proba(X_val, ntree_end=best_iter)[:, 1]

        metrics = {
            "best_iteration": best_iter,
            "train_auc": roc_auc_score(y_train, train_pred),
            "val_auc": roc_auc_score(y_val, val_pred),
        }

        key = (model_cfg.task.type.lower(), model_cfg.task.subtype.lower() if model_cfg.task.subtype else None)
        if key in TASKS_SUPPORTING_THRESHOLDS:
            best_threshold, best_f1 = get_best_f1_thresh(pipeline, X_val, y_val)
            metrics["threshold"] = {
                "value": best_threshold,
                "f1": best_f1
            }

    elif model_cfg.task.type == "forecasting":
        # e.g., Prophet or another time-series model
        forecast_train = model.predict(X_train)
        forecast_val = model.predict(X_val)

        forecast_train = ensure_1d_array(forecast_train)
        forecast_val = ensure_1d_array(forecast_val)

        forecast_train = inverse_transform_target(forecast_train, model_cfg.target.transform)
        forecast_val = inverse_transform_target(forecast_val, model_cfg.target.transform)
        
        metrics = {
            "train_rmse": root_mean_squared_error(y_train, forecast_train),
            "val_rmse": root_mean_squared_error(y_val, forecast_val),
            "train_mae": mean_absolute_error(y_train, forecast_train),
            "val_mae": mean_absolute_error(y_val, forecast_val)
        }

    elif model_cfg.task.type == "regression":
        # Standard regression metrics
        train_pred = pipeline.predict(X_train)
        val_pred = pipeline.predict(X_val)

        train_pred = ensure_1d_array(train_pred)
        val_pred = ensure_1d_array(val_pred)
        
        train_pred = inverse_transform_target(train_pred, model_cfg.target.transform)
        val_pred = inverse_transform_target(val_pred, model_cfg.target.transform)

        metrics = {
            "train_rmse": root_mean_squared_error(y_train, train_pred),
            "val_rmse": root_mean_squared_error(y_val, val_pred),
            "train_mae": mean_absolute_error(y_train, train_pred),
            "val_mae": mean_absolute_error(y_val, val_pred),
            "train_r2": r2_score(y_train, train_pred),
            "val_r2": r2_score(y_val, val_pred)
        }

    else:
        msg = f"Task type {model_cfg.task.type} not supported"
        logging.error(msg)
        raise UserError(msg)

    return metrics