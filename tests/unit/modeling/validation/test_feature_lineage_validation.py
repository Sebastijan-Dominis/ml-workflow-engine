"""Unit tests for feature-lineage validation and construction helpers."""

from __future__ import annotations

import pytest
from ml.exceptions import DataError
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.modeling.validation.feature_lineage import validate_and_construct_feature_lineage

pytestmark = pytest.mark.unit


def _valid_lineage_entry() -> dict[str, str]:
    """Return a minimal valid feature-lineage payload entry."""
    return {
        "name": "booking_context_features",
        "version": "v1",
        "snapshot_id": "2026-03-01T10-00-00",
        "file_hash": "file_hash_abc",
        "in_memory_hash": "memory_hash_abc",
        "feature_schema_hash": "schema_hash_abc",
        "operator_hash": "operator_hash_abc",
        "feature_type": "tabular",
    }


def test_validate_and_construct_feature_lineage_returns_typed_models() -> None:
    """Build typed ``FeatureLineage`` models from valid raw metadata entries."""
    raw_entries = [_valid_lineage_entry(), dict(_valid_lineage_entry(), name="pricing_features")]

    result = validate_and_construct_feature_lineage(raw_entries)

    assert len(result) == 2
    assert all(isinstance(entry, FeatureLineage) for entry in result)
    assert result[0].name == "booking_context_features"
    assert result[1].name == "pricing_features"


def test_validate_and_construct_feature_lineage_wraps_missing_field_errors_as_data_error() -> None:
    """Wrap missing-required-field failures as ``DataError`` with raw-input context."""
    raw_entries = [
        {
            "name": "booking_context_features",
            "version": "v1",
            "snapshot_id": "2026-03-01T10-00-00",
            "file_hash": "file_hash_abc",
            "in_memory_hash": "memory_hash_abc",
            "feature_schema_hash": "schema_hash_abc",
            # operator_hash intentionally missing
            "feature_type": "tabular",
        }
    ]

    with pytest.raises(DataError, match="Error constructing FeatureLineage objects"):
        validate_and_construct_feature_lineage(raw_entries)


def test_validate_and_construct_feature_lineage_wraps_literal_validation_errors_as_data_error() -> None:
    """Wrap invalid literal values instead of leaking pydantic validation exceptions."""
    raw_entries = [dict(_valid_lineage_entry(), feature_type="graph")]

    with pytest.raises(DataError, match="Error constructing FeatureLineage objects"):
        validate_and_construct_feature_lineage(raw_entries)
