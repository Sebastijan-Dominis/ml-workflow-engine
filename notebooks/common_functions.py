import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import random
import math

from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, RocCurveDisplay, roc_auc_score, ConfusionMatrixDisplay, classification_report

def get_data(feature_path):
    X_train = pd.read_parquet(Path(feature_path) / "X_train.parquet")
    X_test = pd.read_parquet(Path(feature_path) / "X_test.parquet")

    y_train = pd.read_parquet(Path(feature_path) / "y_train.parquet")
    y_test = pd.read_parquet(Path(feature_path) / "y_test.parquet")

    return X_train, X_test, y_train, y_test

def search_best_params(pipeline, X_train, y_train, param_distributions, segmented, search_type, cv=3, scoring="roc_auc", error_score=np.nan):
    n_iter = 1
    # if search_type == "broad":
    #     n_iter = 30 if not segmented else 12
    # elif search_type == "narrow":
    #     n_iter = 20 if not segmented else 8
    # else:
    #     raise ValueError("search_type must be either 'broad' or 'narrow'")
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        verbose=2,
        n_jobs=1, # Use 1 to avoid potential GPU memory issues
        random_state=42,
        error_score=error_score
    )

    search.fit(X_train, y_train)

    return search

# Helper function
def refine_int(center, offsets, low, high):
    values = {center}
    for o in offsets:
        values.add(center + o)
        values.add(center - o)
    return sorted(v for v in values if low <= v <= high)

# Helper function
def refine_float_mult(center, factors, low, high, decimals=5):
    values = set()
    for f in factors:
        v = center * f
        if low <= v <= high:
            values.add(round(v, decimals))
    values.add(round(center, decimals))
    return sorted(values)


def prepare_narrow_search_params(best_params):
    narrow_params = {}

    # Tree depth
    if "model__depth" in best_params:
        narrow_params["model__depth"] = refine_int(
            best_params["model__depth"],
            offsets=[1, 2],
            low=2,
            high=12
        )

    # Learning rate
    if "model__learning_rate" in best_params:
        narrow_params["model__learning_rate"] = refine_float_mult(
            best_params["model__learning_rate"],
            factors=[0.7, 0.85, 1.0, 1.15, 1.3],
            low=0.003,
            high=0.5
        )

    # L2 regularization
    if "model__l2_leaf_reg" in best_params:
        narrow_params["model__l2_leaf_reg"] = refine_int(
            best_params["model__l2_leaf_reg"],
            offsets=[1, 2, 3],
            low=1,
            high=30
        )

    # Bagging temperature
    if "model__bagging_temperature" in best_params:
        narrow_params["model__bagging_temperature"] = refine_float_mult(
            best_params["model__bagging_temperature"],
            factors=[0.6, 0.8, 1.0, 1.2, 1.5],
            low=0.0,
            high=5.0
        )

    # Min data in leaf
    if "model__min_data_in_leaf" in best_params:
        narrow_params["model__min_data_in_leaf"] = refine_int(
            best_params["model__min_data_in_leaf"],
            offsets=[2, 5, 10],
            low=1,
            high=50
        )

    # Random strength
    if "model__random_strength" in best_params:
        narrow_params["model__random_strength"] = refine_int(
            best_params["model__random_strength"],
            offsets=[1, 2, 4],
            low=0,
            high=20
        )

    # Freeze GPU-sensitive / discretization params
    if "model__border_count" in best_params:
        narrow_params["model__border_count"] = [
            best_params["model__border_count"]
        ]

    if "model__colsample_bylevel" in best_params:
        narrow_params["model__colsample_bylevel"] = [
            best_params["model__colsample_bylevel"]
        ]

    return narrow_params

# For Jupyter notebooks
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

# For Jupyter notebooks
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

# For python scripts
def get_best_f1_thresh(pipeline, X, y_true):
    y_probs = pipeline.predict_proba(X)[:, 1]

    thresholds = np.linspace(0,1,101)
    f1_scores = []

    for t in thresholds:
        y_pred_thresh = (y_probs >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh))

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

# def get_best_f1_thresh(pipeline, X, y_true):
#     try:
#         y_probs = pipeline.predict_proba(X)[:, 1]

#         thresholds = np.linspace(0,1,101)
#         f1_scores = []

#         for t in thresholds:
#             y_pred_thresh = (y_probs >= t).astype(int)
#             f1_scores.append(f1_score(y_true, y_pred_thresh))

#         best_idx = np.argmax(f1_scores)
#         return thresholds[best_idx]
#     except Exception:
#         logger.exception("Failed to compute best F1 threshold.")
#         raise

def save_training_config(name, task, version, features_path, features_version, target, algorithm, clean_params, iterations, threshold, pipeline_steps):
    validate_schema = pipeline_steps['validate_schema']
    fill_categorical_missing = pipeline_steps['fill_categorical_missing']
    feature_engineering = pipeline_steps['feature_engineering']
    feature_selection = pipeline_steps['feature_selection']

    if task in ['binary_classification', 'regression']:
        explainability = {
            'feature_importance_method': 'PredictionValuesChange',
            'shap_method': 'tree_explainer_mean_abs'
        }
    else:
        explainability = {}

    training_config = {
        'name': name, # e.g., "cancellation_city_hotel_online_ta", "cancellation_global"
        'task': task, # e.g., "binary_classification", "regression"
        'version': version, # e.g., "v1", "v2"

        'data': {
            "features_path": features_path, # e.g., "data/features/cancellation/global/v1/", "data/features/cancellation/hotel_market_segment/city_hotel_online_ta_v1"
            "features_version": features_version, # e.g., "v1", "v2"
            "target": target, # e.g., "is_canceled", "no_show"
            "train_file": "X_train.parquet",
            "val_file": "X_val.parquet",
            "test_file": "X_test.parquet",
            "y_train": "y_train.parquet",
            "y_val": "y_val.parquet",
            "y_test": "y_test.parquet"
        },

        'model': {
            'algorithm': algorithm, # e.g., "catboost"
            'params': {
                **clean_params, # cleaned hyperparameters from search
                'iterations': iterations, # increase iterations for final training
                'task_type': 'CPU',    # switch to CPU for final model
                'random_state': 42, # reproducibility
                'verbose': 100 # print progress every 100 iterations
            },
            'threshold': threshold
        },

        'pipeline': {
            'validate_schema': validate_schema,
            'fill_categorical_missing': fill_categorical_missing,
            'feature_engineering': feature_engineering,
            'feature_selection': feature_selection
        },

        'explainability': explainability
    }

    config_path = Path(f"ml/training/train_configs/{training_config['name']}_{training_config['version']}.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as file:
        yaml.safe_dump(training_config, file, sort_keys=False)