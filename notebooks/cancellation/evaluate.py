import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    disp_train = ConfusionMatrixDisplay.from_predictions(
        y_train, y_pred_train,
        display_labels=["Not canceled", "Canceled"],
        cmap="Blues",
        normalize=None
    )
    disp_train.ax_.set_title("Training Set Confusion Matrix")
    plt.show();

    print(classification_report(y_train, y_pred_train, target_names=["Not canceled", "Canceled"]))
    
    disp_test = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_test,
        display_labels=["Not canceled", "Canceled"],
        cmap="Blues",
        normalize=None
    )
    disp_test.ax_.set_title("Test Set Confusion Matrix")
    plt.show();

    print(classification_report(y_test, y_pred_test, target_names=["Not canceled", "Canceled"]))

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