"""Classification evaluation helpers.

This module contains utilities to load a trained classifier, read feature
parquet datasets, compute standard classification metrics and evaluate the
model across train/validation/test splits.

Key functions:
- `evaluate_classification`: high-level entrypoint used by the evaluation
    runner to compute metrics and return a structured result mapping.
"""

# General imports
import importlib
import pandas as pd
import numpy as np
import joblib
import logging
logger = logging.getLogger(__name__)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pathlib import Path
from typing import Optional, Protocol

# Utility imports
from ml.runners.evaluation.utils import assert_keys

# Define a Protocol for classifiers with predict_proba method
class ProbabilisticClassifier(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

def get_file_paths(model_configs: dict) -> tuple[Path, Path]:
    """Resolve artifact and features paths from model configuration.

    Args:
        model_configs (dict): Model configuration mapping containing
            `artifacts.model` and `features.path` entries.

    Returns:
        tuple[pathlib.Path, pathlib.Path]: `(model_file, features_folder)`.
    """

    # Get file paths
    model_file = Path(model_configs["artifacts"]["model"])
    features_folder = Path(model_configs["features"]["path"])

    # Return paths and threshold
    return model_file, features_folder

def load_model(model_file: Path, model_configs: dict) -> ProbabilisticClassifier:
    """Load a serialized model from disk.

    Some model classes or custom objects may live in `ml.components.*` and
    must be imported prior to joblib deserialization. This function performs
    that import and returns the loaded estimator.

    Args:
        model_file (pathlib.Path): Path to the serialized model file.
        model_configs (dict): Model configuration used to determine the
            deserialization import path.

    Returns:
        ProbabilisticClassifier: Deserialized model/estimator.
    """

    # Import necessary components for deserialization
    importlib.import_module(f"ml.components.{model_configs['name']}_{model_configs['version']}")

    # Load the model
    with open(model_file, "rb") as f:
        model = joblib.load(f)
    
    # Return the loaded model
    return model


def get_data_splits(features_folder: Path) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Load train/validation/test feature and label parquet files.

    Args:
        features_folder (pathlib.Path): Directory containing the parquet files
            named `X_train.parquet`, `y_train.parquet`, etc.

    Returns:
        dict: Mapping of split name to a tuple `(X, y)` where `X` is a
            DataFrame and `y` is a pandas Series.
    """

    # Load data
    X_train = pd.read_parquet(features_folder / "X_train.parquet")
    y_train = pd.read_parquet(features_folder / "y_train.parquet")
    X_val = pd.read_parquet(features_folder / "X_val.parquet")
    y_val = pd.read_parquet(features_folder / "y_val.parquet")
    X_test = pd.read_parquet(features_folder / "X_test.parquet")
    y_test = pd.read_parquet(features_folder / "y_test.parquet")

    # Ensure y are 1D arrays
    y_train = y_train.iloc[:, 0]
    y_val = y_val.iloc[:, 0]
    y_test = y_test.iloc[:, 0]



    # Create data splits
    data_splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

    # Return data splits
    return data_splits

def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: Optional[pd.Series] = None) -> dict[str, float]:
    """Compute commonly used classification metrics.

    Args:
        y_true (pd.Series): Ground-truth binary labels.
        y_pred (pd.Series): Binary predictions.
        y_prob (Optional[pd.Series], optional): Predicted probabilities for the positive class.

    Returns:
        dict: Metric values for `accuracy`, `f1`, and `roc_auc` (or `None`
            when `roc_auc` is not defined for the provided inputs).
    """

    # Compute basic classification metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    # Compute ROC AUC, handle case where it's not applicable
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob if y_prob is not None else [])
    except ValueError:
        logger.warning("ROC AUC score is not defined for the provided inputs.")
        metrics["roc_auc"] = None

    # Return computed metrics
    return metrics

def evaluate_split(model: ProbabilisticClassifier, X: pd.DataFrame, y: pd.Series, best_threshold: float = 0.5) -> dict[str, float]: # default threshold=0.5
    """Evaluate a single split and return metrics.

    Args:
        model: Fitted classifier implementing `predict_proba`.
        X (DataFrame): Feature matrix for the split.
        y (Series): True labels for the split.
        best_threshold (float): Probability threshold for converting
            predicted probabilities into binary labels.

    Returns:
        dict: Metrics computed for the split.
    """

    # Predict probabilities for the positive class
    y_prob_arr = model.predict_proba(X)[:, 1]

    y_prob = pd.Series(
        y_prob_arr,
        index=y.index,
        name="y_prob",
    )

    # Convert probabilities to binary predictions based on the threshold
    y_pred = pd.Series(
        (y_prob >= best_threshold).astype(int),
        index=y.index,
        name="y_pred",
    )


    # Compute and return metrics
    return compute_metrics(y, y_pred, y_prob)

def evaluate_model(model: ProbabilisticClassifier, data_splits: dict[str, tuple[pd.DataFrame, pd.Series]], best_threshold: float = 0.5) -> dict[str, dict[str, float]]:
    """Evaluate `model` across all provided data splits.

    Args:
        model: Fitted classifier.
        data_splits (dict): Mapping of split name to `(X, y)` tuple.
        best_threshold (float): Threshold for converting probabilities to
            binary predictions.

    Returns:
        dict: Mapping from split name to computed metrics dictionary.
    """

    # Create a dictionary to hold evaluation results
    evaluation_results = {}

    # Evaluate each data split
    for split_name, (X, y) in data_splits.items():
        metrics = evaluate_split(model, X, y, best_threshold=best_threshold)
        evaluation_results[split_name] = metrics

    #  Return evaluation results
    return evaluation_results

def evaluate_classification(model_configs: dict, best_threshold: float) -> dict[str, dict[str, float]]:
    """High-level evaluation entrypoint for binary classification tasks.

    Args:
        model_configs (dict): Model configuration mapping.
        best_threshold (float): Probability threshold to use when
            converting probabilities to binary labels.

    Returns:
        dict: Nested mapping of split names to metric dictionaries.
    """

    # Step 1 - Ensure required keys are present
    assert_keys(model_configs["artifacts"], ["model"])
    assert_keys(model_configs["features"], ["path"])

    # Step 2 - Get file paths
    model_file, features_folder = get_file_paths(model_configs)

    # Step 3 - Load model
    model = load_model(model_file, model_configs)

    # Step 4 - Get data splits
    data_splits = get_data_splits(features_folder)

    # Step 5 - Evaluate the model
    evaluation_results = evaluate_model(model, data_splits, best_threshold=best_threshold)

    # Step 6 - Log success message
    logger.info(f"Evaluation completed successfully for model '{model_configs['name']}_{model_configs['version']}'.")

    # Step 7 - Return evaluation results
    return evaluation_results