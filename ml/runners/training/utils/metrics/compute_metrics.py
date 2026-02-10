from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.registry.tasks_supporting_thresholds import TASKS_SUPPORTING_THRESHOLDS
from ml.runners.training.utils.metrics.best_f1 import get_best_f1_thresh


def compute_metrics(*, model: Any, pipeline: Pipeline, model_cfg: TrainModelConfig, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame) -> dict[str, float]:
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
        
        metrics = {
            "train_rmse": mean_squared_error(y_train, forecast_train, squared=False),
            "val_rmse": mean_squared_error(y_val, forecast_val, squared=False),
            "train_mae": mean_absolute_error(y_train, forecast_train),
            "val_mae": mean_absolute_error(y_val, forecast_val)
        }

    else:
        raise NotImplementedError(f"Task type {model_cfg.task.type} not supported")

    return metrics