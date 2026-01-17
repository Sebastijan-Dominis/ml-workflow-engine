import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pathlib import Path

# ------------------------------------------
# Helper function to get file paths
# ------------------------------------------
def get_file_paths(model_configs):
    # Get file paths
    model_file = Path(model_configs["artifacts"]["model"])
    features_folder = Path(model_configs["features"]["path"])

    # Return paths and threshold
    return model_file, features_folder

# ------------------------------------------
# Helper function to load the model
# ------------------------------------------
def load_model(model_file):
    # Import necessary components for deserialization
    from ml.components.cancellation_v1 import (
        SchemaValidator,
        FillCategoricalMissing,
        FeatureEngineer,
        FeatureSelector
    )

    # Load the model
    with open(model_file, "rb") as f:
        model = joblib.load(f)
    
    # Return the loaded model
    return model

# ------------------------------------------
# Helper function to get data splits
# ------------------------------------------
def get_data_splits(features_folder):
    # Load data
    X_train = pd.read_parquet(features_folder / "X_train.parquet")
    y_train = pd.read_parquet(features_folder / "y_train.parquet")
    X_val = pd.read_parquet(features_folder / "X_val.parquet")
    y_val = pd.read_parquet(features_folder / "y_val.parquet")
    X_test = pd.read_parquet(features_folder / "X_test.parquet")
    y_test = pd.read_parquet(features_folder / "y_test.parquet")

    # Ensure y are 1D arrays
    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()


    # Create data splits
    data_splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

    # Return data splits
    return data_splits

# ------------------------------------------
# Helper function to compute metrics
# ------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    # Compute basic classification metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    # Compute ROC AUC, handle case where it's not applicable
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = None

    # Return computed metrics
    return metrics

# ------------------------------------------
# Helper function to evaluate one data split
# ------------------------------------------
def evaluate_split(model, X, y, best_threshold=0.5): # default threshold=0.5
    # Predict probabilities for the positive class
    y_prob = model.predict_proba(X)[:, 1]

    # Convert probabilities to binary predictions based on the threshold
    y_pred = (y_prob >= best_threshold).astype(int)

    # Compute and return metrics
    return compute_metrics(y, y_pred, y_prob)

# ------------------------------------------
# Helper function to evaluate the whole model
# ------------------------------------------
def evaluate_model(model, data_splits, best_threshold=0.5):
    # Create a dictionary to hold evaluation results
    evaluation_results = {}

    # Evaluate each data split
    for split_name, (X, y) in data_splits.items():
        metrics = evaluate_split(model, X, y, best_threshold=best_threshold)
        evaluation_results[split_name] = metrics

    #  Return evaluation results
    return evaluation_results

# ------------------------------------------
# Main evaluation function
# ------------------------------------------
def evaluate_classification(model_configs, best_threshold):
    # Step 1 - Get file paths
    model_file, features_folder = get_file_paths(model_configs)

    # Step 2 - Load model
    model = load_model(model_file)

    # Step 3 - Get data splits
    data_splits = get_data_splits(features_folder)

    # Step 4 - Evaluate the model
    evaluation_results = evaluate_model(model, data_splits, best_threshold=best_threshold)

    # Step 5 - Print success message
    print("Evaluation completed successfully.")

    # Step 6 - Return evaluation results
    return evaluation_results