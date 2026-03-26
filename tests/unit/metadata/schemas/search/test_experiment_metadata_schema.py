"""Unit tests for the ExperimentMetadata schema model.

These tests validate strict enum/literal behavior, nested split requirements,
and optional target-transform/class-weighting payload handling.
"""

import importlib
import sys
import types

import pytest
from ml.exceptions import ConfigError
from pydantic import ValidationError

# ExperimentMetadata imports ml.types package, which imports catboost at import time.
if "catboost" not in sys.modules:
    catboost_stub = types.ModuleType("catboost")
    catboost_stub.__dict__.update(
        {
            "CatBoostClassifier": type("CatBoostClassifier", (), {}),
            "CatBoostRegressor": type("CatBoostRegressor", (), {}),
        }
    )
    sys.modules["catboost"] = catboost_stub

ExperimentMetadata = importlib.import_module("ml.search.models.experiment_metadata").ExperimentMetadata


pytestmark = pytest.mark.unit


def _experiment_metadata_payload() -> dict:
    """Return a valid ExperimentMetadata payload baseline."""

    return {
        "problem": "cancellation",
        "segment": "city_hotel_online_ta",
        "version": "v1",
        "experiment_id": "exp-001",
        "sources": {"main": "defaults", "extends": ["segment"]},
        "env": "dev",
        "best_params_path": "experiments/cancellation/best_params.json",
        "algorithm": "catboost",
        "pipeline_version": "v1",
        "created_by": "tests",
        "created_at": "2026-03-05T12:00:00",
        "owner": "ml-team",
        "feature_lineage": [
            {
                "name": "booking_context_features",
                "version": "v1",
                "snapshot_id": "20260305T120000_abcd",
                "file_hash": "file-hash",
                "in_memory_hash": "mem-hash",
                "feature_schema_hash": "schema-hash",
                "operator_hash": "op-hash",
                "feature_type": "tabular",
                "file_name": "features.py",
                "data_format": "parquet",
            }
        ],
        "seed": 42,
        "hardware": {"task_type": "CPU", "devices": []},
        "git_commit": "abc123",
        "config_hash": "cfg-hash",
        "validation_status": "ok",
        "pipeline_hash": "pipe-hash",
        "scoring_method": "roc_auc",
        "splits_info": {
            "train": {"n_rows": 1000, "class_distribution": {"0": 700, "1": 300}, "positive_rate": 0.3},
            "val": {"n_rows": 200, "class_distribution": {"0": 140, "1": 60}, "positive_rate": 0.3},
            "test": {"n_rows": 200, "class_distribution": {"0": 140, "1": 60}, "positive_rate": 0.3},
        },
        "target_transform": None,
        "class_weighting": None,
    }


def test_experiment_metadata_accepts_valid_payload() -> None:
    """Ensure ExperimentMetadata accepts a complete valid payload."""

    result = ExperimentMetadata.model_validate(_experiment_metadata_payload())

    assert result.experiment_id == "exp-001"
    assert result.hardware.task_type == "CPU"


def test_experiment_metadata_rejects_invalid_env_literal() -> None:
    """Ensure ExperimentMetadata enforces env Literal values."""

    payload = _experiment_metadata_payload()
    payload["env"] = "staging"

    with pytest.raises(ValidationError, match="env"):
        ExperimentMetadata.model_validate(payload)


def test_experiment_metadata_rejects_invalid_feature_lineage_type() -> None:
    """Ensure ExperimentMetadata enforces allowed feature_type literals."""

    payload = _experiment_metadata_payload()
    payload["feature_lineage"][0]["feature_type"] = "image"

    with pytest.raises(ValidationError, match="feature_type"):
        ExperimentMetadata.model_validate(payload)


def test_experiment_metadata_rejects_missing_split_block() -> None:
    """Ensure ExperimentMetadata requires train/val/test split info blocks."""

    payload = _experiment_metadata_payload()
    del payload["splits_info"]["val"]

    with pytest.raises(ValidationError, match="val"):
        ExperimentMetadata.model_validate(payload)


def test_experiment_metadata_accepts_optional_target_transform_payload() -> None:
    """Ensure ExperimentMetadata accepts a valid nested target transform payload."""

    payload = _experiment_metadata_payload()
    payload["target_transform"] = {"enabled": True, "type": "log1p", "lambda_value": None}

    result = ExperimentMetadata.model_validate(payload)

    assert result.target_transform is not None
    assert result.target_transform.type == "log1p"


def test_experiment_metadata_rejects_invalid_target_transform_lambda_usage() -> None:
    """Ensure ExperimentMetadata propagates transform lambda consistency checks."""

    payload = _experiment_metadata_payload()
    payload["target_transform"] = {"enabled": True, "type": "sqrt", "lambda_value": 0.3}

    with pytest.raises(ConfigError, match="lambda_value should only be provided"):
        ExperimentMetadata.model_validate(payload)


def test_experiment_metadata_rejects_invalid_class_weighting_policy() -> None:
    """Ensure ExperimentMetadata enforces class-weighting policy literals."""

    payload = _experiment_metadata_payload()
    payload["class_weighting"] = {
        "policy": "sometimes",
        "imbalance_threshold": 0.25,
        "strategy": "balanced",
    }

    with pytest.raises(ValidationError, match="policy"):
        ExperimentMetadata.model_validate(payload)
