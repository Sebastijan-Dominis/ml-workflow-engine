import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, f1_score, roc_auc_score)


def evaluate_binary_classifier(pipeline, X_train, y_train, X_test, y_test, positive_label, negative_label):
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
