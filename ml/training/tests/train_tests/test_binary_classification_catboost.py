"""Integration and unit tests for CatBoost-based binary classification trainer.

Contains a small smoke test that runs training against prepared
features and a negative test ensuring training fails when feature
files are missing.
"""

import pytest

from ml.training.train_scripts.custom_training_scripts.binary_classification_catboost import (
    train_binary_classification_with_catboost
)

def cancellation_test_cfg(features_path: str) -> dict:
    """Return a test configuration for the cancellation binary classification model."""
    return {
        "name": "test_model",
        "task": "binary_classification",
        "version": "v0",
        "data": {
            "features_path": features_path,
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
                "iterations": 5,
                "depth": 3,
            },
            "threshold": 0.5,
        },
        "pipeline": {
            "validate_schema": True,
            "fill_categorical_missing": True,
            "feature_engineering": True,
            "feature_selection": True,
        },
        "explainability": {
            "feature_importance_method": "gain",
            "shap_method": "tree",
        },
    }

@pytest.mark.integration
@pytest.mark.slow
def test_catboost_trainer_smoke() -> None:
    """
    Smoke test: ensure training runs end-to-end on real data and returns
    a fitted sklearn Pipeline with an expected `model` step.
    """

    cfg = cancellation_test_cfg(features_path="data/features/cancellation/global/v1")

    pipeline = train_binary_classification_with_catboost(
        name_version="cancellation_city_hotel_online_ta_v1",  # must exist in ml.components
        cfg=cfg,
    )

    assert pipeline is not None
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")
    assert "model" in pipeline.named_steps
    assert pipeline.named_steps["model"].__class__.__name__ == "CatBoostClassifier"

def test_training_fails_on_missing_feature_file() -> None:
    """Verify trainer raises when the configured feature path does not exist."""

    cfg = cancellation_test_cfg(features_path="does/not/exist")
    with pytest.raises(Exception):
        train_binary_classification_with_catboost(
            name_version="cancellation_city_hotel_online_ta_v1",
            cfg=cfg,
        )