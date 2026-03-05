"""Unit tests for feature-freezing metadata schema.

These tests validate that FreezeMetadata accepts complete payloads and rejects
invalid nested runtime/data-lineage shapes.
"""

import importlib
import sys
import types

import pytest
from pydantic import ValidationError

# Freeze metadata imports ml.types, which imports catboost at module import time.
if "catboost" not in sys.modules:
    catboost_stub = types.ModuleType("catboost")
    catboost_stub.__dict__.update(
        {
            "CatBoostClassifier": type("CatBoostClassifier", (), {}),
            "CatBoostRegressor": type("CatBoostRegressor", (), {}),
        }
    )
    sys.modules["catboost"] = catboost_stub

FreezeMetadata = importlib.import_module("ml.metadata.schemas.features.feature_freezing").FreezeMetadata


pytestmark = pytest.mark.unit


def _lineage_entry_payload() -> dict:
    """Return a valid DataLineageEntry-compatible dictionary payload."""

    return {
        "ref": "data/processed",
        "name": "hotel_bookings",
        "version": "v1",
        "format": "parquet",
        "path_suffix": "data.parquet",
        "merge_key": "row_id",
        "snapshot_id": "20260305T120000_abcd",
        "path": "data/processed/hotel_bookings/v1/data.parquet",
        "loader_validation_hash": "loader-hash",
        "data_hash": "data-hash",
        "row_count": 1200,
        "column_count": 30,
    }


def _runtime_payload() -> dict:
    """Return a valid FreezeRuntimeInfo payload."""

    return {
        "git_commit": "abc123",
        "runtime_info": {
            "os": "Windows",
            "os_release": "11",
            "architecture": "AMD64",
            "processor": "Intel",
            "ram_total_gb": 16.0,
            "platform_string": "Windows-11-10.0.22631-SP0",
            "hostname": "test-host",
            "python_version": "3.11.9",
            "python_impl": "CPython",
            "python_build": ["main", "Mar 1 2026"],
        },
        "deps": {
            "numpy": "1.26.4",
            "pandas": "2.2.0",
            "scikit_learn": "1.4.0",
            "pyarrow": "15.0.0",
            "pydantic": "2.10.0",
            "PyYAML": "6.0.1",
        },
        "python_executable": "C:/Python311/python.exe",
    }


def _freeze_metadata_payload() -> dict:
    """Return a complete valid FreezeMetadata payload."""

    return {
        "created_by": "tests",
        "created_at": "2026-03-05T12:00:00",
        "owner": "ml-team",
        "feature_type": "tabular",
        "snapshot_path": "feature_store/booking_context_features/v1/snapshot.parquet",
        "snapshot_id": "20260305T120000_abcd",
        "schema_path": "feature_store/booking_context_features/v1/schema.yaml",
        "data_lineage": [_lineage_entry_payload()],
        "in_memory_hash": "mem-hash",
        "file_hash": "file-hash",
        "operator_hash": "op-hash",
        "config_hash": "cfg-hash",
        "feature_schema_hash": "schema-hash",
        "runtime": _runtime_payload(),
        "row_count": 1200,
        "column_count": 40,
        "duration_seconds": 3.14,
    }


def test_freeze_metadata_accepts_valid_payload() -> None:
    """Ensure FreezeMetadata accepts a complete valid payload."""

    result = FreezeMetadata.model_validate(_freeze_metadata_payload())

    assert result.snapshot_id == "20260305T120000_abcd"
    assert result.row_count == 1200
    assert result.runtime.deps.PyYAML == "6.0.1"


def test_freeze_metadata_rejects_missing_dependency_field() -> None:
    """Ensure FreezeMetadata rejects malformed runtime dependency sections."""

    payload = _freeze_metadata_payload()
    del payload["runtime"]["deps"]["PyYAML"]

    with pytest.raises(ValidationError, match="PyYAML"):
        FreezeMetadata.model_validate(payload)


def test_freeze_metadata_rejects_invalid_data_lineage_entry_shape() -> None:
    """Ensure FreezeMetadata rejects incomplete data-lineage entry payloads."""

    payload = _freeze_metadata_payload()
    del payload["data_lineage"][0]["loader_validation_hash"]

    with pytest.raises(ValidationError, match="loader_validation_hash"):
        FreezeMetadata.model_validate(payload)
