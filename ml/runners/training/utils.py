import logging
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from pathlib import Path

def load_train_and_val_data(cfg_model_specs: dict) -> tuple:
    """Load training and validation features and labels from disk.

    Args:
        cfg_model_specs (dict): Configuration dictionary with keys under ``features``:
            - ``path``: base folder containing feature files.
            - ``X_train``, ``X_val``: parquet files for X.
            - ``y_train``, ``y_val``: parquet files for labels.

    Returns:
        tuple: ``(X_train, y_train, X_val, y_val)`` as pandas DataFrames/Series.

    Raises:
        Any exception encountered while reading files is logged and re-raised
        so callers can handle or fail the training run explicitly.
    """

    try:
        features_path = Path(cfg_model_specs["features"]["path"])

        X_train = pd.read_parquet(features_path / cfg_model_specs["features"]["X_train"])
        X_val = pd.read_parquet(features_path / cfg_model_specs["features"]["X_val"])

        y_train = pd.read_parquet(features_path / cfg_model_specs["features"]["y_train"])
        y_val = pd.read_parquet(features_path / cfg_model_specs["features"]["y_val"])

        return X_train, y_train, X_val, y_val
    except Exception:
        logger.exception("Error loading data")
        raise

def get_best_f1_thresh(pipeline, X, y_true):
    y_probs = pipeline.predict_proba(X)[:, 1]

    thresholds = np.linspace(0,1,101)
    f1_scores = []

    for t in thresholds:
        y_pred_thresh = (y_probs >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh))

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])