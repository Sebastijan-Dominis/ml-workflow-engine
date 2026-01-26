import pandas as pd

from pathlib import Path

def get_data(feature_path):
    X_train = pd.read_parquet(Path(feature_path) / "X_train.parquet")
    X_test = pd.read_parquet(Path(feature_path) / "X_test.parquet")

    y_train = pd.read_parquet(Path(feature_path) / "y_train.parquet")
    y_test = pd.read_parquet(Path(feature_path) / "y_test.parquet")

    return X_train, X_test, y_train, y_test