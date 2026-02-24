import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import PipelineContractError, UserError
from ml.runners.evaluation.constants.data_splits import DataSplits
from ml.runners.evaluation.utils.get_row_ids import get_row_ids

logger = logging.getLogger(__name__)

def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: Optional[pd.Series] = None) -> dict[str, float]:
    # Compute basic classification metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    # Compute ROC AUC, handle case where it's not applicable
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            logger.warning("ROC AUC score is not defined for the provided inputs.")
            metrics["roc_auc"] = "undefined"
    else:
        metrics["roc_auc"] = "undefined"

    # Return computed metrics
    return metrics

def evaluate_split(
    pipeline: Pipeline, 
    X: pd.DataFrame, 
    y: pd.Series, 
    *, 
    split_row_ids: pd.Series,
    split_name: str, 
    best_threshold: float | None
) -> tuple[dict[str, float], pd.DataFrame]: # default threshold=0.5
    # Predict probabilities for the positive class
    probs = pipeline.predict_proba(X)
    if probs.shape[1] < 2:
        msg = f"Expected predict_proba output to have at least 2 columns for binary classification. Got shape: {probs.shape}"
        logger.error(msg)
        raise PipelineContractError(msg)
    y_prob = probs[:, 1]

    # Convert probabilities to binary predictions based on the threshold
    y_pred = pd.Series(
        (y_prob >= best_threshold).astype(int),
        index=y.index,
        name="y_pred",
    )

    y_prob = pd.Series(y_prob, index=y.index, name="y_prob")

    # Compute and return metrics
    metrics = compute_metrics(y, y_pred, y_prob)

    df_preds = pd.DataFrame({
        "row_id": split_row_ids,
        "split": split_name,
        "y_true": y,
        "y_pred": y_pred,
        "y_proba": y_prob,
    })
    
    return metrics, df_preds

def evaluate_model(model_cfg: TrainModelConfig, *, pipeline: Pipeline, data_splits: DataSplits, best_threshold: float | None) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    # Create a dictionary to hold evaluation results
    evaluation_metrics = {}
    prediction_dfs = {}

    # Evaluate each data split
    for split_name, (X, y) in data_splits.__dict__.items():
        if model_cfg.task.subtype and model_cfg.task.subtype.lower() == "binary":
            split_row_ids = get_row_ids(X)
            if "row_id" in X.columns:
                X = X.drop(columns=["row_id"])
            logger.debug(f"Evaluating split '{split_name}' with best_threshold={best_threshold} and {len(y)} samples.")
            metrics, df_preds = evaluate_split(
                pipeline=pipeline, 
                X=X, 
                y=y, 
                split_row_ids=split_row_ids, 
                split_name=split_name, 
                best_threshold=best_threshold
            )
            evaluation_metrics[split_name] = metrics
            prediction_dfs[split_name] = df_preds
        elif model_cfg.task.subtype and model_cfg.task.subtype.lower() == "multiclass":
            msg = f"Multiclass classification evaluation is not yet implemented for task '{model_cfg.task.type}' with subtype '{model_cfg.task.subtype}'."
            logger.error(msg)
            raise UserError(msg)
        else:
            msg = f"Unsupported task subtype '{model_cfg.task.subtype}' for evaluation."
            logger.error(msg)
            raise PipelineContractError(msg)

    #  Return evaluation results
    return evaluation_metrics, prediction_dfs