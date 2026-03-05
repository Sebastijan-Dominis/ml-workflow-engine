"""Unit tests for data metadata schema models.

These tests validate that raw/interim/processed metadata schemas accept well-formed
payloads and reject structurally incomplete payloads in critical nested sections.
"""

import pytest
from ml.metadata.schemas.data.interim import InterimDatasetMetadata
from ml.metadata.schemas.data.processed import ProcessedDatasetMetadata
from ml.metadata.schemas.data.raw import RawSnapshotMetadata
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _shared_interim_processed_payload() -> dict:
    """Return a valid base payload shared by interim and processed schemas."""

    return {
        "rows": 1200,
        "columns": {
            "count": 3,
            "names": ["hotel", "lead_time", "is_canceled"],
            "dtypes": {
                "hotel": "category",
                "lead_time": "int16",
                "is_canceled": "int8",
            },
        },
        "created_at": "2026-03-05T12:00:00",
        "created_by": "tests",
        "owner": "ml-team",
        "source_data": {
            "name": "hotel_bookings_raw",
            "version": "v1",
            "format": "csv",
            "snapshot_id": "20260305T120000_abcd",
            "path": "data/raw/hotel_bookings/v1/data.csv",
        },
        "data": {
            "name": "hotel_bookings",
            "version": "v2",
            "output": {
                "path_suffix": "data.parquet",
                "format": "parquet",
                "compression": "snappy",
            },
            "hash": "abc123",
        },
        "memory": {
            "old_memory_mb": 100.0,
            "new_memory_mb": 80.0,
            "change_mb": -20.0,
            "change_percentage": -20.0,
        },
        "config_hash": "cfg123",
        "duration": 2.5,
        "runtime_info": {
            "pandas_version": "2.2.0",
            "numpy_version": "1.26.4",
            "yaml_version": "6.0.1",
            "python_version": "3.11.9",
        },
    }


def test_raw_snapshot_metadata_accepts_valid_payload() -> None:
    """Ensure raw snapshot metadata model accepts a complete valid payload."""

    payload = {
        "rows": 1000,
        "columns": {
            "count": 2,
            "names": ["hotel", "lead_time"],
            "dtypes": {"hotel": "category", "lead_time": "int16"},
        },
        "created_at": "2026-03-05T12:00:00",
        "created_by": "tests",
        "owner": "ml-team",
        "data": {
            "name": "hotel_bookings",
            "version": "v1",
            "format": "csv",
            "path_suffix": "data.csv",
            "hash": "abc123",
        },
        "memory_usage_mb": 120.5,
        "raw_run_id": "raw-run-001",
    }

    result = RawSnapshotMetadata.model_validate(payload)

    assert result.rows == 1000
    assert result.data.path_suffix == "data.csv"
    assert result.raw_run_id == "raw-run-001"


def test_raw_snapshot_metadata_rejects_missing_required_field() -> None:
    """Ensure raw snapshot metadata rejects payloads missing required fields."""

    payload = {
        "rows": 1000,
        "columns": {"count": 1, "names": ["hotel"], "dtypes": {"hotel": "category"}},
        "created_at": "2026-03-05T12:00:00",
        "created_by": "tests",
        "owner": "ml-team",
        "data": {
            "name": "hotel_bookings",
            "version": "v1",
            "format": "csv",
            "path_suffix": "data.csv",
            "hash": "abc123",
        },
        "memory_usage_mb": 120.5,
    }

    with pytest.raises(ValidationError, match="raw_run_id"):
        RawSnapshotMetadata.model_validate(payload)


def test_interim_dataset_metadata_accepts_valid_payload() -> None:
    """Ensure interim dataset metadata model accepts a complete shared payload."""

    payload = _shared_interim_processed_payload()
    payload["interim_run_id"] = "interim-run-001"

    result = InterimDatasetMetadata.model_validate(payload)

    assert result.interim_run_id == "interim-run-001"
    assert result.data.hash == "abc123"
    assert result.source_data.snapshot_id == "20260305T120000_abcd"


def test_processed_dataset_metadata_accepts_row_id_info_when_present() -> None:
    """Ensure processed metadata supports optional row-id information payloads."""

    payload = _shared_interim_processed_payload()
    payload["processed_run_id"] = "processed-run-001"
    payload["row_id_info"] = {
        "cols_for_row_id": ["hotel", "lead_time"],
        "fingerprint": "row-fp-123",
    }

    result = ProcessedDatasetMetadata.model_validate(payload)

    assert result.processed_run_id == "processed-run-001"
    assert result.row_id_info is not None
    assert result.row_id_info.fingerprint == "row-fp-123"


def test_processed_dataset_metadata_accepts_none_row_id_info() -> None:
    """Ensure processed metadata accepts omitted optional row-id section."""

    payload = _shared_interim_processed_payload()
    payload["processed_run_id"] = "processed-run-002"

    result = ProcessedDatasetMetadata.model_validate(payload)

    assert result.processed_run_id == "processed-run-002"
    assert result.row_id_info is None


def test_processed_dataset_metadata_rejects_invalid_row_id_info_shape() -> None:
    """Ensure processed metadata rejects malformed row-id structures."""

    payload = _shared_interim_processed_payload()
    payload["processed_run_id"] = "processed-run-003"
    payload["row_id_info"] = {
        "cols_for_row_id": ["hotel", "lead_time"],
    }

    with pytest.raises(ValidationError, match="fingerprint"):
        ProcessedDatasetMetadata.model_validate(payload)
