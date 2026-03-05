"""Unit tests for the InterimConfig schema in ml.data.config.schemas.interim. The tests verify that the schema correctly accepts valid payloads, assigns default invariants, and raises ConfigError for invalid raw_data_version formats and invariant values that violate policy constraints. Additionally, there is a test to ensure that allowed values for hotel types are validated against the allowed registry."""
import pytest
from ml.data.config.schemas.interim import InterimConfig, Invariants
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


def _valid_interim_payload() -> dict:
    """Helper function to create a valid payload for testing the InterimConfig schema.

    Returns:
        dict: A valid payload for testing the InterimConfig schema.
    """
    return {
        "data": _data_info_payload(),
        "data_schema": {},
        "raw_data_version": "v1",
        "cleaning": {},
        "invariants": {},
        "lineage": _lineage_payload(),
    }


def test_interim_config_accepts_valid_payload_and_assigns_default_invariants() -> None:
    """Test that the InterimConfig schema accepts a valid payload and correctly assigns default invariants for lead_time and arrival_date_week_number."""
    config = InterimConfig.model_validate(_valid_interim_payload())

    assert config.raw_data_version == "v1"
    assert config.invariants.lead_time is not None
    assert config.invariants.lead_time.min is not None
    assert config.invariants.lead_time.min.value == 0


def test_interim_config_rejects_invalid_raw_data_version_format() -> None:
    """Test that the InterimConfig schema raises a ConfigError if the raw_data_version does not follow the expected format (e.g., "v1", "v2", etc.)."""
    payload = _valid_interim_payload()
    payload["raw_data_version"] = "version_1"

    with pytest.raises(ConfigError, match="Invalid raw_data_version"):
        InterimConfig.model_validate(payload)


def test_invariants_reject_min_lower_than_policy_constraint() -> None:
    """Test that the Invariants schema raises a ConfigError if the minimum value for lead_time is set to a value lower than the policy constraint (e.g., less than 0)."""
    with pytest.raises(ConfigError, match="min value -1.0 is less than 0"):
        Invariants.model_validate({"lead_time": {"min": {"value": -1, "op": "gte"}}})


def test_invariants_reject_max_higher_than_policy_constraint() -> None:
    """Test that the Invariants schema raises a ConfigError if the maximum value for arrival_date_week_number is set to a value higher than the policy constraint (e.g., greater than 53)."""
    with pytest.raises(ConfigError, match="max value 54.0 is greater than 53"):
        Invariants.model_validate({"arrival_date_week_number": {"max": {"value": 54, "op": "lte"}}})


def test_invariants_reject_values_outside_allowed_registry() -> None:
    """Test that the Invariants schema raises a ConfigError if the allowed values for hotel are set to a value that is not in the allowed registry (e.g., "Hostel" instead of "Resort", "City Hotel", or "Resort Hotel")."""
    with pytest.raises(ConfigError, match="allowed values"):
        Invariants.model_validate({"hotel": {"allowed_values": ["Hostel"]}})
