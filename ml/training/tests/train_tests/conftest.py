"""Pytest fixtures used by training tests.

Provides lightweight, minimal configuration dictionaries suitable for
unit tests that do not require actual feature files.
"""

import pytest


@pytest.fixture
def minimal_training_cfg(tmp_path: str) -> dict:
    """Return a minimal training configuration mapping for tests.

    The fixture uses `tmp_path` for safe dummy file locations and
    contains the smallest set of keys required by the training code.
    """
    return {
        "name": "test_model",
        "task": "binary_classification",
        "version": "v0",
        "data": {
            "features_path": str(tmp_path),  # safe dummy path
            "features_version": "test",
            "target": "label",
            "train_file": "X_train.parquet",
            "val_file": "X_val.parquet",
            "test_file": "X_test.parquet",
            "y_train": "y_train.parquet",
            "y_val": "y_val.parquet",
            "y_test": "y_test.parquet",
        },
        "model": {
            "algorithm": "catboost",
            "params": {
                "iterations": 1,
                "depth": 2,
            },
            "threshold": 0.5,
        },
        "pipeline": {
            "validate_schema": False,
            "fill_categorical_missing": False,
            "feature_engineering": False,
            "feature_selection": False,
        },
        "explainability": {
            "feature_importance_method": "gain",
            "shap_method": "tree",
        },
    }