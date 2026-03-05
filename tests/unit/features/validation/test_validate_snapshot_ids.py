"""Unit tests for feature lineage snapshot-id consistency validation."""

from dataclasses import dataclass

import pytest
from ml.exceptions import DataError
from ml.features.validation.validate_snapshot_ids import validate_snapshot_ids

pytestmark = pytest.mark.unit


@dataclass
class _LineageItem:
    """Minimal lineage stub carrying snapshot id used by validator."""

    snapshot_id: str


def test_validate_snapshot_ids_passes_when_expected_and_actual_ids_match() -> None:
    """Accept snapshot lineage when requested and loaded IDs match in order."""
    snapshot_selection = [{"snapshot_id": "s1"}, {"snapshot_id": "s2"}]
    feature_lineage = [_LineageItem(snapshot_id="s1"), _LineageItem(snapshot_id="s2")]

    validate_snapshot_ids(feature_lineage, snapshot_selection)


def test_validate_snapshot_ids_raises_when_expected_snapshot_ids_are_missing() -> None:
    """Reject validation when expected snapshot selection is empty."""
    with pytest.raises(DataError, match="Missing snapshot IDs"):
        validate_snapshot_ids([_LineageItem(snapshot_id="s1")], [])


def test_validate_snapshot_ids_raises_when_actual_snapshot_ids_are_missing() -> None:
    """Reject validation when loaded feature lineage list is empty."""
    with pytest.raises(DataError, match="Missing snapshot IDs"):
        validate_snapshot_ids([], [{"snapshot_id": "s1"}])


def test_validate_snapshot_ids_raises_on_order_or_value_mismatch() -> None:
    """Reject lineage when actual snapshot IDs differ from requested selection."""
    snapshot_selection = [{"snapshot_id": "s1"}, {"snapshot_id": "s2"}]
    feature_lineage = [_LineageItem(snapshot_id="s2"), _LineageItem(snapshot_id="s1")]

    with pytest.raises(DataError, match="do not match the expected snapshot IDs"):
        validate_snapshot_ids(feature_lineage, snapshot_selection)
