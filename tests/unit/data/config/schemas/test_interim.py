"""Unit tests for interim data configuration schemas."""
import pytest
from ml.data.config.schemas.interim import InterimConfig, Invariants
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


def _data_info_payload() -> dict:
    """Return a valid data metadata payload for tests."""
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
    """Return a valid lineage payload for tests."""
    return {
        "created_by": "tests",
        "created_at": "2026-03-05T00:00:00",
    }


def _valid_interim_payload() -> dict:
    """Return a valid `InterimConfig` payload."""
    return {
        "data": _data_info_payload(),
        "data_schema": {},
        "raw_data_version": "v1",
        "cleaning": {},
        "invariants": {},
        "lineage": _lineage_payload(),
    }


def test_interim_config_accepts_valid_payload_and_assigns_default_invariants() -> None:
    """Verify valid payload parsing and default invariant assignment."""
    config = InterimConfig.model_validate(_valid_interim_payload())

    assert config.raw_data_version == "v1"
    assert config.invariants.lead_time is not None
    assert config.invariants.lead_time.min is not None
    assert config.invariants.lead_time.min.value == 0


def test_interim_config_rejects_invalid_raw_data_version_format() -> None:
    """Verify rejection of invalid `raw_data_version` formats."""
    payload = _valid_interim_payload()
    payload["raw_data_version"] = "version_1"

    with pytest.raises(ConfigError, match="Invalid raw_data_version"):
        InterimConfig.model_validate(payload)


def test_invariants_reject_min_lower_than_policy_constraint() -> None:
    """Verify rejection of invariant minimum values below policy constraints."""
    with pytest.raises(ConfigError, match="min value -1.0 is less than 0"):
        Invariants.model_validate({"lead_time": {"min": {"value": -1, "op": "gte"}}})


def test_invariants_reject_max_higher_than_policy_constraint() -> None:
    """Verify rejection of invariant maximum values above policy constraints."""
    with pytest.raises(ConfigError, match="max value 54.0 is greater than 53"):
        Invariants.model_validate({"arrival_date_week_number": {"max": {"value": 54, "op": "lte"}}})


def test_invariants_reject_values_outside_allowed_registry() -> None:
    """Verify rejection of allowed-values entries outside the policy registry."""
    with pytest.raises(ConfigError, match="allowed values"):
        Invariants.model_validate({"hotel": {"allowed_values": ["Hostel"]}})
