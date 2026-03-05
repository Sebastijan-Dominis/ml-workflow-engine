"""Unit tests for cross-field validators in the ModelSpecs schema."""

import pytest
from ml.config.schemas.model_specs import ModelSpecs
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


def _base_model_specs_payload() -> dict:
    """Helper function to generate a valid base payload for ModelSpecs tests.

    Args:
        None

    Returns:
        dict: A valid payload dictionary for ModelSpecs.
    """
    return {
        "problem": "cancellation",
        "segment": {"name": "city_hotel_online_ta"},
        "version": "v1",
        "task": {"type": "classification", "subtype": "binary"},
        "target": {
            "name": "is_canceled",
            "version": "v1",
            "allowed_dtypes": ["int8"],
            "classes": {
                "count": 2,
                "positive_class": 1,
                "min_class_count": 10,
            },
            "transform": {
                "enabled": False,
                "type": None,
                "lambda_value": None,
            },
        },
        "segmentation": {
            "enabled": False,
            "include_in_model": False,
            "filters": [],
        },
        "min_rows": 100,
        "split": {
            "strategy": "random",
            "stratify_by": "is_canceled",
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42,
        },
        "algorithm": "catboost",
        "model_class": "CatBoostClassifier",
        "pipeline": {
            "version": "v1",
            "path": "ml/pipelines/tabular",
        },
        "scoring": {
            "policy": "fixed",
            "fixed_metric": "roc_auc",
            "pr_auc_threshold": None,
        },
        "class_weighting": {
            "policy": "off",
            "imbalance_threshold": None,
            "strategy": None,
        },
        "feature_store": {
            "path": "feature_store",
            "feature_sets": [
                {
                    "name": "booking_context_features",
                    "version": "v1",
                    "schema_format": "yaml",
                    "input_schema": "input.yaml",
                    "derived_schema": "derived.yaml",
                    "data_format": "parquet",
                    "file_name": "data.parquet",
                }
            ],
        },
        "explainability": {
            "enabled": True,
            "top_k": 20,
            "methods": {
                "feature_importances": {"enabled": False, "type": None},
                "shap": {"enabled": False, "approximate": None},
            },
        },
        "data_type": "tabular",
        "model_specs_lineage": {
            "created_by": "tests",
            "created_at": "2026-03-05T00:00:00",
        },
    }


def test_model_specs_accepts_valid_classification_payload() -> None:
    """Test that the ModelSpecs schema accepts a valid payload for a classification task.

    Args:
        None

    Returns:
        None
    """
    cfg = ModelSpecs.model_validate(_base_model_specs_payload())

    assert cfg.task.type == "classification"
    assert cfg.target.classes is not None
    assert cfg.class_weighting.policy == "off"


def test_model_specs_rejects_classification_without_classes() -> None:
    """Test that the ModelSpecs schema raises a ConfigError if classes are not provided for a classification task.

    Args:
        None

    Returns:
        None
    """
    payload = _base_model_specs_payload()
    payload["target"]["classes"] = None

    with pytest.raises(ConfigError, match="Classes must be provided for classification tasks"):
        ModelSpecs.model_validate(payload)


def test_model_specs_rejects_regression_when_classes_are_provided() -> None:
    """Test that the ModelSpecs schema raises a ConfigError if classes are provided for a regression task.

    Args:
        None

    Returns:
        None
    """
    payload = _base_model_specs_payload()
    payload["task"] = {"type": "regression", "subtype": None}

    with pytest.raises(ConfigError, match="Classes should not be provided for task type"):
        ModelSpecs.model_validate(payload)


def test_model_specs_rejects_target_transform_for_non_regression_tasks() -> None:
    """Test that the ModelSpecs schema raises a ConfigError if target transformation is enabled for non-regression tasks.

    Args:
        None

    Returns:
        None
    """
    payload = _base_model_specs_payload()
    payload["target"]["transform"] = {
        "enabled": True,
        "type": "log1p",
        "lambda_value": None,
    }

    with pytest.raises(ConfigError, match="Target transformation is only applicable for regression tasks"):
        ModelSpecs.model_validate(payload)


def test_model_specs_rejects_regression_transform_enabled_without_type() -> None:
    """Test that the ModelSpecs schema raises a ConfigError if target transformation is enabled for regression tasks but type is not specified.

    Args:
        None

    Returns:
        None
    """
    payload = _base_model_specs_payload()
    payload["task"] = {"type": "regression", "subtype": None}
    payload["target"]["classes"] = None
    payload["target"]["transform"] = {
        "enabled": True,
        "type": None,
        "lambda_value": None,
    }

    with pytest.raises(ConfigError, match="Target transformation type must be specified"):
        ModelSpecs.model_validate(payload)


def test_model_specs_rejects_class_weighting_for_non_classification_tasks() -> None:
    """Test that the ModelSpecs schema raises a ConfigError if class weighting is configured for non-classification tasks.

    Args:
        None

    Returns:
        None
    """
    payload = _base_model_specs_payload()
    payload["task"] = {"type": "regression", "subtype": None}
    payload["target"]["classes"] = None
    payload["class_weighting"] = {
        "policy": "always",
        "imbalance_threshold": 0.2,
        "strategy": "balanced",
    }

    with pytest.raises(ConfigError, match="Class weighting is only applicable for classification tasks"):
        ModelSpecs.model_validate(payload)
