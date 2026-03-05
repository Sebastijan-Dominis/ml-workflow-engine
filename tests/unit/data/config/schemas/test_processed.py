"""Unit tests for the ProcessedConfig schema in ml.data.config.schemas.processed. The tests verify that the schema correctly assigns default values for sensitive columns to remove, raises ConfigError for invalid interim_data_version formats, and ensures that the lineage information is properly validated. The tests use helper functions to create valid payloads for data information and lineage to facilitate testing of the ProcessedConfig schema."""
import pytest
from ml.data.config.schemas.processed import ProcessedConfig
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


def _data_info_payload() -> dict:
    """Helper function to create a valid data information payload for testing.

    Returns:
        dict: A valid data information payload.
    """
    return {
        "name": "hotel_bookings",
        "version": "v1",
        "output": {
            "path_suffix": "data.parquet",
            "format": "parquet",
            "compression": "snappy",
        },
    }


def _lineage_payload() -> dict:
    """Helper function to create a valid lineage payload for testing.

    Returns:
        dict: A valid lineage payload.
    """
    return {
        "created_by": "tests",
        "created_at": "2026-03-05T00:00:00",
    }


def test_processed_config_defaults_sensitive_remove_columns() -> None:
    """Test that the ProcessedConfig schema assigns the default list of sensitive columns to remove if not provided in the payload."""
    config = ProcessedConfig.model_validate(
        {
            "data": _data_info_payload(),
            "interim_data_version": "v2",
            "lineage": _lineage_payload(),
        }
    )

    assert config.interim_data_version == "v2"
    assert config.remove_columns == ["name", "email", "phone_number", "credit_card"]


def test_processed_config_rejects_invalid_interim_data_version_format() -> None:
    """Test that the ProcessedConfig schema raises a ConfigError if the interim_data_version does not follow the expected format (e.g., "v1", "v2", etc.)."""
    with pytest.raises(ConfigError, match="Invalid interim_data_version"):
        ProcessedConfig.model_validate(
            {
                "data": _data_info_payload(),
                "interim_data_version": "2",
                "lineage": _lineage_payload(),
            }
        )
