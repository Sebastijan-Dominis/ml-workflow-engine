"""Metric computation and split evaluation helpers for classification tasks."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import PipelineContractError, UserError
from ml.runners.evaluation.constants.data_splits import DataSplits
from ml.runners.evaluation.utils.get_row_ids import get_row_ids

logger = logging.getLogger(__name__)

def expected_calibration_error(
    y_true: pd.Series,
    y_prob: pd.Series,
    n_bins: int = 10,
) -> float:
    """Compute expected calibration error (ECE) from labels and probabilities.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of confidence bins.

    Returns:
        float: Expected calibration error value.
    """

    y_true_np = np.asarray(y_true)
    y_prob_np = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_np, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue

        bin_accuracy = y_true_np[mask].mean()
        bin_confidence = y_prob_np[mask].mean()
        bin_weight = mask.mean()

        ece += np.abs(bin_accuracy - bin_confidence) * bin_weight

    return float(ece)

def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: Optional[pd.Series] = None,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute threshold-based and probability-based classification metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Optional predicted probabilities for the positive class.
        threshold: Optional decision threshold used for converting probabilities.

    Returns:
        dict[str, float]: Computed classification metrics.
    """

    metrics: dict[str, float] = {}

    # -------------------------
    # Base rates
    # -------------------------
    metrics["positive_rate"] = float(y_true.mean())
    metrics["predicted_positive_rate"] = float(y_pred.mean())
    if threshold is not None:
        metrics["threshold"] = float(threshold)

    # -------------------------
    # Threshold-based metrics
    # -------------------------
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    # Specificity (true negative rate)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        logger.warning("Confusion matrix could not be computed.")
        metrics["specificity"] = np.nan

    # -------------------------
    # Probability-based metrics
    # -------------------------
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            logger.warning("ROC AUC undefined.")
            metrics["roc_auc"] = np.nan

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            logger.warning("PR AUC undefined.")
            metrics["pr_auc"] = np.nan

        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except ValueError:
            logger.warning("Log loss undefined.")
            metrics["log_loss"] = np.nan

        try:
            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        except ValueError:
            logger.warning("Brier score undefined.")
            metrics["brier_score"] = np.nan

        # Calibration
        metrics["ece"] = expected_calibration_error(y_true, y_prob)

    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["log_loss"] = np.nan
        metrics["brier_score"] = np.nan
        metrics["ece"] = np.nan

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
    """Evaluate a single split and return metrics plus prediction dataframe.

    Args:
        pipeline: Fitted model pipeline.
        X: Split features.
        y: Split target labels.
        split_row_ids: Row identifiers corresponding to split records.
        split_name: Split label (train/val/test).
        best_threshold: Decision threshold for positive-class prediction.

    Returns:
        tuple[dict[str, float], pd.DataFrame]: Split metrics and prediction rows.
    """

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
    logger.info(f"Computing classification metrics for the {split_name} split...")
    metrics = compute_metrics(y, y_pred, y_prob, threshold=best_threshold)

    df_preds = pd.DataFrame({
        "row_id": split_row_ids,
        "split": split_name,
        "y_true": y,
        "y_pred": y_pred,
        "y_proba": y_prob,
    })
    
    return metrics, df_preds

def evaluate_model(
    model_cfg: TrainModelConfig, 
    *, 
    pipeline: Pipeline, 
    data_splits: DataSplits, 
    best_threshold: float | None
) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]:
    """Evaluate all configured splits and aggregate metrics/prediction frames.

    Args:
        model_cfg: Training configuration used for task/subtype decisions.
        pipeline: Fitted model pipeline.
        data_splits: Data split container.
        best_threshold: Decision threshold for positive-class prediction.

    Returns:
        tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame]]: Metrics and predictions by split.
    """

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