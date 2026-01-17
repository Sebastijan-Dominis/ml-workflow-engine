import joblib
import yaml
import json
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def get_model_configs():
    with open("configs/models.yaml") as f:
        configs = yaml.safe_load(f)

    return configs["cancellation_v1"]

# ------------------------------------------
# Helper function to get all file paths
# ------------------------------------------
def get_file_paths(model_configs):
    metadata_file = Path(model_configs["artifacts"]["metadata"])
    model_file = Path(model_configs["artifacts"]["model"])
    features_folder = Path(model_configs["features"]["path"])

    return metadata_file, model_file, features_folder

# ------------------------------------------
# Helper function to load metadata
# ------------------------------------------
def load_metadata(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata

# ------------------------------------------
# Helper function to load the model
# ------------------------------------------
def load_model(model_file):
    from ml.components.cancellation_v1 import (
        SchemaValidator,
        FillCategoricalMissing,
        FeatureEngineer,
        FeatureSelector
    )

    with open(model_file, "rb") as f:
        model = joblib.load(f)
    
    return model

# ------------------------------------------
# Helper function to get data splits
# ------------------------------------------
def get_data_splits(features_folder):
    # ------------------------------------------
    # Step 1 — Load data
    # ------------------------------------------
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


    # ------------------------------------------
    # Step 2 — Create data splits
    # ------------------------------------------
    data_splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

    return data_splits

# ------------------------------------------
# Helper function to compute metrics
# ------------------------------------------
def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = None

    return metrics

# ------------------------------------------
# Helper function to evaluate model
# ------------------------------------------
def evaluate_model(model, X, y, threshold=0.5): # default threshold=0.5
    y_prob = model.predict_proba(X)[:, 1] # Probability for the positive class
    y_pred = (y_prob >= threshold).astype(int)

    return compute_metrics(y, y_pred, y_prob)

# ------------------------------------------
# Helper function to update metadata with evaluation results
# ------------------------------------------
def update_metadata(metadata, data_splits, model, best_threshold):
    metadata.setdefault("metrics", {})
    for split_name, (X, y) in data_splits.items():
        metrics = evaluate_model(model, X, y, threshold=best_threshold)
        metadata["metrics"][split_name] = metrics
    metadata["threshold"] = best_threshold

# ------------------------------------------
# Helper function to save metadata
# ------------------------------------------
def save_metadata(metadata, metadata_file):
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

# ------------------------------------------
# Main evaluation script
# ------------------------------------------
def main():
    model_configs = get_model_configs()

    metadata_file, model_file, features_folder = get_file_paths(model_configs)

    metadata = load_metadata(metadata_file)
    model = load_model(model_file)
    data_splits = get_data_splits(features_folder)
    best_threshold = model_configs.get("threshold", 0.5)

    update_metadata(metadata, data_splits, model, best_threshold)

    save_metadata(metadata, metadata_file)
        
if __name__ == "__main__":
    main()