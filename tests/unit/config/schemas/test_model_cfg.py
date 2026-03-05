"""Unit tests for top-level search/train model config schemas."""

import pytest
from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _base_common_payload() -> dict:
    """Helper function to create a valid base payload for testing the SearchModelConfig and TrainModelConfig schemas. This payload includes all the required fields for both schemas, allowing individual tests to modify or extend it as needed without having to redefine the common structure.

    Returns:
        dict: A valid base payload dictionary for testing the SearchModelConfig and TrainModelConfig schemas
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
            "transform": {"enabled": False, "type": None, "lambda_value": None},
        },
        "segmentation": {"enabled": False, "include_in_model": False, "filters": []},
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
        "pipeline": {"version": "v1", "path": "ml/pipelines/tabular"},
        "scoring": {"policy": "fixed", "fixed_metric": "roc_auc", "pr_auc_threshold": None},
        "class_weighting": {"policy": "off", "imbalance_threshold": None, "strategy": None},
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
        "model_specs_lineage": {"created_by": "tests", "created_at": "2026-03-05T00:00:00"},
    }


def _search_payload() -> dict:
    """Helper function to create a valid payload for testing the SearchModelConfig schema. This payload extends the base common payload with additional fields specific to the search configuration, such as search parameters and lineage information for the search process.

    Returns:
        dict: A valid payload dictionary for testing the SearchModelConfig schema.
    """
    payload = _base_common_payload()
    payload.update(
        {
            "search": {
                "random_state": 42,
                "broad": {"iterations": 2, "n_iter": 5, "param_distributions": {}},
            },
            "seed": 42,
            "cv": 5,
            "search_lineage": {"created_by": "tests", "created_at": "2026-03-05T00:00:00"},
        }
    )
    return payload


def _train_payload() -> dict:
    """Helper function to create a valid payload for testing the TrainModelConfig schema. This payload extends the base common payload with additional fields specific to the training configuration, such as training parameters and lineage information for the training process.

    Returns:
        dict: A valid payload dictionary for testing the TrainModelConfig schema.
    """
    payload = _base_common_payload()
    payload.update(
        {
            "training": {"iterations": 200},
            "seed": 42,
            "cv": 5,
            "training_lineage": {"created_by": "tests", "created_at": "2026-03-05T00:00:00"},
        }
    )
    return payload


def test_search_model_config_accepts_valid_payload_and_defaults() -> None:
    """Test that the SearchModelConfig schema accepts a valid payload and correctly assigns default values for optional fields."""
    cfg = SearchModelConfig.model_validate(_search_payload())

    assert cfg.seed == 42
    assert cfg.cv == 5
    assert cfg.extends == []
    assert cfg.verbose == 100
    assert cfg.search.hardware.task_type == "GPU"


def test_train_model_config_accepts_valid_payload_and_defaults() -> None:
    """Test that the TrainModelConfig schema accepts a valid payload and correctly assigns default values for optional fields."""
    cfg = TrainModelConfig.model_validate(_train_payload())

    assert cfg.seed == 42
    assert cfg.cv == 5
    assert cfg.extends == []
    assert cfg.verbose == 100
    assert cfg.training.hardware.task_type == "CPU"


def test_search_model_config_forbids_unknown_top_level_fields() -> None:
    """Test that the SearchModelConfig schema raises a ValidationError when the payload contains unknown top-level fields that are not defined in the schema."""
    payload = _search_payload()
    payload["unexpected_field"] = "not-allowed"

    with pytest.raises(ValidationError, match="unexpected_field"):
        SearchModelConfig.model_validate(payload)


def test_train_model_config_forbids_unknown_top_level_fields() -> None:
    """Test that the TrainModelConfig schema raises a ValidationError when the payload contains unknown top-level fields that are not defined in the schema."""
    payload = _train_payload()
    payload["unexpected_field"] = "not-allowed"

    with pytest.raises(ValidationError, match="unexpected_field"):
        TrainModelConfig.model_validate(payload)


def test_search_model_config_allows_optional_training_stub_field() -> None:
    """Test that the SearchModelConfig schema allows an optional training field in the payload, which can be used as a stub for training configuration when the search configuration is being validated independently. The test adds a training field to the search payload and verifies that it is accepted without causing validation errors."""
    payload = _search_payload()
    payload["training"] = {"inherited": "from-defaults"}

    cfg = SearchModelConfig.model_validate(payload)

    assert cfg.training == {"inherited": "from-defaults"}


def test_train_model_config_allows_optional_search_stub_field() -> None:
    """Test that the TrainModelConfig schema allows an optional search field in the payload, which can be used as a stub for search configuration when the training configuration is being validated independently. The test adds a search field to the training payload and verifies that it is accepted without causing validation errors."""
    payload = _train_payload()
    payload["search"] = {"inherited": "from-defaults"}

    cfg = TrainModelConfig.model_validate(payload)

    assert cfg.search == {"inherited": "from-defaults"}
