"""Notebook utility helpers for binary classification evaluation.

This module provides plotting and scoring helpers commonly used during model
exploration in notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, f1_score, roc_auc_score)


def evaluate_binary_classifier(pipeline, X_train, y_train, X_test, y_test, positive_label, negative_label):
    """Evaluate a binary classifier on train and test sets.

    The function computes predictions, prints classification reports, plots
    confusion matrices for train and test splits, and plots ROC curves with AUC
    summaries.

    Args:
        pipeline: Fitted estimator or pipeline with ``predict`` and
            ``predict_proba`` methods.
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_test: Test feature matrix.
        y_test: Test target vector.
        positive_label: Display label for the positive class.
        negative_label: Display label for the negative class.

    Returns:
        None

    Raises:
        AttributeError: If ``pipeline`` does not implement required prediction
            methods.

    Side Effects:
        Prints classification summaries and renders multiple matplotlib figures
        (confusion matrices and ROC curves).
    """
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    disp_train = ConfusionMatrixDisplay.from_predictions(
        y_train, y_pred_train,
        display_labels=[negative_label, positive_label],
        cmap="Blues",
        normalize=None
    )
    disp_train.ax_.set_title("Training Set Confusion Matrix")
    plt.show();

    print(classification_report(y_train, y_pred_train, target_names=[negative_label, positive_label]))
    
    disp_test = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_test,
        display_labels=[negative_label, positive_label],
        cmap="Blues",
        normalize=None
    )
    disp_test.ax_.set_title("Test Set Confusion Matrix")
    plt.show();

    print(classification_report(y_test, y_pred_test, target_names=[negative_label, positive_label]))

    y_train_probs = pipeline.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_train_probs)
    print(f"Training Set ROC AUC: {auc_train}")

    RocCurveDisplay.from_predictions(y_train, y_train_probs)
    plt.show();

    y_probs_test = pipeline.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_probs_test)
    print(f"Test Set ROC AUC: {auc_test}")

    RocCurveDisplay.from_predictions(y_test, y_probs_test)
    plt.show();

def optimal_f1_search(pipeline, X, y_true):
    """Search for the probability threshold that maximizes F1 score.

    The function evaluates thresholds on a fixed grid from ``0.00`` to ``1.00``
    in ``0.01`` increments, reports the best threshold and F1 score, and plots
    F1 versus threshold.

    Args:
        pipeline: Fitted estimator or pipeline with a ``predict_proba`` method.
        X: Feature matrix for threshold search.
        y_true: True binary labels corresponding to ``X``.

    Returns:
        float: Threshold that yields the maximum F1 score on the provided data.
    """
    
    y_probs = pipeline.predict_proba(X)[:, 1]

    thresholds = np.linspace(0,1,101)
    f1_scores = []

    for t in thresholds:
        y_pred_thresh = (y_probs >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh))

    best_idx = np.argmax(f1_scores)
    print("Best threshold:", thresholds[best_idx])
    print("Best F1:", f1_scores[best_idx])

    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Decision Threshold")
    plt.show()

    return thresholds[best_idx]
