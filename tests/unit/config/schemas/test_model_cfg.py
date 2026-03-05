"""Unit tests for top-level search/train model config schemas."""

import pytest
from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _base_common_payload() -> dict:
    """Return a valid shared payload for model-config schema tests."""
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
    """Return a valid payload for `SearchModelConfig` tests."""
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
    """Return a valid payload for `TrainModelConfig` tests."""
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
    """Verify valid search payload parsing and default assignment."""
    cfg = SearchModelConfig.model_validate(_search_payload())

    assert cfg.seed == 42
    assert cfg.cv == 5
    assert cfg.extends == []
    assert cfg.verbose == 100
    assert cfg.search.hardware.task_type == "GPU"


def test_train_model_config_accepts_valid_payload_and_defaults() -> None:
    """Verify valid training payload parsing and default assignment."""
    cfg = TrainModelConfig.model_validate(_train_payload())

    assert cfg.seed == 42
    assert cfg.cv == 5
    assert cfg.extends == []
    assert cfg.verbose == 100
    assert cfg.training.hardware.task_type == "CPU"


def test_search_model_config_forbids_unknown_top_level_fields() -> None:
    """Verify that unknown top-level fields are rejected for search config."""
    payload = _search_payload()
    payload["unexpected_field"] = "not-allowed"

    with pytest.raises(ValidationError, match="unexpected_field"):
        SearchModelConfig.model_validate(payload)


def test_train_model_config_forbids_unknown_top_level_fields() -> None:
    """Verify that unknown top-level fields are rejected for train config."""
    payload = _train_payload()
    payload["unexpected_field"] = "not-allowed"

    with pytest.raises(ValidationError, match="unexpected_field"):
        TrainModelConfig.model_validate(payload)


def test_search_model_config_allows_optional_training_stub_field() -> None:
    """Verify that search config accepts an optional training stub field."""
    payload = _search_payload()
    payload["training"] = {"inherited": "from-defaults"}

    cfg = SearchModelConfig.model_validate(payload)

    assert cfg.training == {"inherited": "from-defaults"}


def test_train_model_config_allows_optional_search_stub_field() -> None:
    """Verify that train config accepts an optional search stub field."""
    payload = _train_payload()
    payload["search"] = {"inherited": "from-defaults"}

    cfg = TrainModelConfig.model_validate(payload)

    assert cfg.search == {"inherited": "from-defaults"}
