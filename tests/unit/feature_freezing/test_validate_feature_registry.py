"""Unit tests for the validate_feature_registry function in ml.feature_freezing.freeze_strategies.config.validate_feature_registry. The tests verify that the function correctly validates raw configurations for feature registries, returns properly typed configuration objects, and raises appropriate errors for unsupported data types. The tests use helper functions to create valid raw configurations for tabular features to facilitate testing of the validation logic."""
import pytest
from ml.exceptions import UserError
from ml.feature_freezing.freeze_strategies.config.validate_feature_registry import (
    validate_feature_registry,
)
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

pytestmark = pytest.mark.unit


def _valid_tabular_raw_config() -> dict:
    return {
        "type": "tabular",
        "description": "unit-test config",
        "data": [
            {
                "name": "hotel_bookings",
                "version": "v1",
                "format": "parquet",
            }
        ],
        "feature_store_path": "feature_store/booking_context_features/v1",
        "columns": ["country", "lead_time"],
        "feature_roles": {
            "categorical": ["country"],
            "numerical": ["lead_time"],
            "datetime": [],
        },
        "constraints": {
            "forbid_nulls": ["country"],
            "max_cardinality": {"country": 200},
        },
        "storage": {
            "format": "parquet",
        },
        "lineage": {
            "created_by": "tests",
            "created_at": "2026-03-05T00:00:00",
        },
    }


def test_validate_feature_registry_returns_typed_tabular_config() -> None:
    """Test that the validate_feature_registry function returns a properly typed TabularFeaturesConfig when given a valid raw configuration for tabular features."""
    config = validate_feature_registry(
        raw_config=_valid_tabular_raw_config(),
        data_type="tabular",
    )

    assert isinstance(config, TabularFeaturesConfig)
    assert config.type == "tabular"
    assert config.columns == ["country", "lead_time"]


def test_validate_feature_registry_wraps_error_for_unsupported_data_type() -> None:
    """Test that the validate_feature_registry function raises a UserError with an appropriate message when given an unsupported data type, and that the original error is preserved as the cause.
    """
    with pytest.raises(UserError, match="Feature registry validation failed") as error:
        validate_feature_registry(raw_config={}, data_type="image")

    assert isinstance(error.value.__cause__, UserError)
    assert "Unsupported data type: image" in str(error.value.__cause__)
