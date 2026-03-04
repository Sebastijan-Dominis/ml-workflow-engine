"""Threshold search utility for maximizing F1 in binary classification."""

import logging

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

def get_best_f1_threshold(pipeline, X, y_true):
    """Find probability threshold in [0, 1] grid that maximizes F1 score.

    Args:
        pipeline: Fitted probabilistic classifier pipeline.
        X: Feature matrix used to generate predicted probabilities.
        y_true: Ground-truth binary labels.

    Returns:
        Tuple containing best threshold and corresponding F1 score.
    """

    y_probs = pipeline.predict_proba(X)[:, 1]

    thresholds = np.linspace(0,1,101)
    f1_scores = []

    for t in thresholds:
        y_pred_thresh = (y_probs >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh))

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])
