"""Tests for the classification evaluation helpers.

These unit tests validate data loading, model deserialization, metric
computation and the top-level `evaluate_classification` orchestration.
"""

import pandas as pd
import numpy as np
import joblib
import pytest

from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ml.training.evaluation_scripts.custom_evaluation_scripts.evaluate_classification import (
    get_file_paths,
    load_model,
    get_data_splits,
    compute_metrics,
    evaluate_split,
    evaluate_model,
    evaluate_classification,
)

def test_get_file_paths(tmp_path: Path) -> None:
    """Test that get_file_paths correctly resolves paths from config."""

    # Create dummy config
    cfg = {
        "artifacts": {
            "model": str(tmp_path / "model.joblib"),
        },
        "features": {
            "path": str(tmp_path / "features"),
        },
    }

    model_file, features_folder = get_file_paths(cfg)

    assert model_file == tmp_path / "model.joblib"
    assert features_folder == tmp_path / "features"

def test_load_model(tmp_path: Path) -> None:
    """Test that load_model correctly loads a serialized sklearn Pipeline."""
    # Use standard sklearn pipeline, no custom classes
    model = Pipeline([
        ("classifier", DummyClassifier())
    ])

    # Serialize the pipeline
    model_file = tmp_path / "model.joblib"
    joblib.dump(model, model_file)

    # Dummy config
    cfg = {
        "name": "dummy_model",
        "version": "v1",
    }

    # Load the model
    loaded_model = load_model(model_file, cfg)

    # Check that it's a pipeline
    from sklearn.pipeline import Pipeline as SKPipeline
    assert isinstance(loaded_model, SKPipeline)

    # Check that last step is a DummyClassifier
    final_step = loaded_model.steps[-1][1]
    from sklearn.dummy import DummyClassifier as SKDummy
    assert isinstance(final_step, SKDummy)

def test_get_data_splits(tmp_path: Path) -> None:
    """Test that get_data_splits correctly loads data splits."""
    # Create dummy feature and label data
    X_train = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    y_train = pd.DataFrame([0, 1])
    X_val = pd.DataFrame({"feature1": [5], "feature2": [6]})
    y_val = pd.DataFrame([1])
    X_test = pd.DataFrame({"feature1": [7], "feature2": [8]})
    y_test = pd.DataFrame([0])

    # Save to parquet files
    X_train.to_parquet(tmp_path / "X_train.parquet")
    y_train.to_parquet(tmp_path / "y_train.parquet")
    X_val.to_parquet(tmp_path / "X_val.parquet")
    y_val.to_parquet(tmp_path / "y_val.parquet")
    X_test.to_parquet(tmp_path / "X_test.parquet")
    y_test.to_parquet(tmp_path / "y_test.parquet")

    features_folder = tmp_path

    data_splits = get_data_splits(features_folder)

    assert "train" in data_splits
    assert "val" in data_splits
    assert "test" in data_splits

    X_train_loaded, y_train_loaded = data_splits["train"]
    pd.testing.assert_frame_equal(X_train, X_train_loaded)
    pd.testing.assert_series_equal(y_train.iloc[:, 0], y_train_loaded)

def test_compute_metrics_basic() -> None:
    """Test that compute_metrics returns expected metrics for binary classification."""
    y_true = pd.Series([0,1,1,0])
    y_pred = pd.Series([0,1,0,0])
    y_prob = pd.Series([0.1,0.9,0.4,0.2])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in metrics and "f1" in metrics and "roc_auc" in metrics

def test_evaluate_split_returns_metrics() -> None:
    """Test that evaluate_split returns a metrics dictionary."""
    class DummyModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]])
    
    X = pd.DataFrame({"feat":[1,2,3]})
    y = pd.Series([0,1,0])
    metrics = evaluate_split(DummyModel(), X, y, best_threshold=0.5)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics

def test_evaluate_model(tmp_path: Path) -> None:
    """Test that evaluate_model returns metrics for all splits."""
    class DummyModel:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]])
    
    X_train = pd.DataFrame({"feat":[1,2,3]})
    y_train = pd.Series([0,1,0])
    X_val = pd.DataFrame({"feat":[4,5,6]})
    y_val = pd.Series([1,0,1])
    data_splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val)
    }
    metrics = evaluate_model(DummyModel(), data_splits, best_threshold=0.5)
    assert "train" in metrics and "val" in metrics

@pytest.mark.integration
def test_evaluate_classification(tmp_path: Path) -> None:
    """Integration test: end-to-end evaluation on dummy data and model."""
    # Dummy data
    X_train = pd.DataFrame({"feat":[1,2,3]})
    y_train = pd.DataFrame([0,1,0])
    X_val = pd.DataFrame({"feat":[4,5,6]})
    y_val = pd.DataFrame([1,0,1])
    X_test = pd.DataFrame({"feat":[7,8,9]})
    y_test = pd.DataFrame([0,1,0])
    features_folder = tmp_path
    X_train.to_parquet(features_folder / "X_train.parquet")
    y_train.to_parquet(features_folder / "y_train.parquet")
    X_val.to_parquet(features_folder / "X_val.parquet")
    y_val.to_parquet(features_folder / "y_val.parquet")
    X_test.to_parquet(features_folder / "X_test.parquet")
    y_test.to_parquet(features_folder / "y_test.parquet")

    # Create and fit dummy model
    pipeline = Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    pipeline.fit(X_train, y_train.values.ravel())
    model_file = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_file)

    # Dummy config
    cfg = {
        "artifacts": {"model": str(model_file)},
        "features": {"path": str(features_folder)},
        "name": "dummy_model",
        "version": "v1",
    }

    # Run evaluation
    results = evaluate_classification(cfg, 0.5)
    assert "train" in results
    assert "val" in results
    assert "test" in results

def test_evaluate_classification_missing_model_raises() -> None:
    """Test that evaluate_classification raises KeyError if model path is missing."""
    cfg = {"artifacts": {}, "features": {}}
    with pytest.raises(KeyError):
        evaluate_classification(cfg, 0.5)
