"""Unit tests for stage-wise dataframe memory delta computation."""

import pytest
from ml.data.utils.memory.compute_memory_change import compute_memory_change
from ml.exceptions import DataError

pytestmark = pytest.mark.unit


def test_compute_memory_change_for_interim_stage_uses_root_memory_usage() -> None:
    """Read baseline memory from interim metadata root and compute deltas."""
    metadata = {"memory_usage_mb": 100.0}

    result = compute_memory_change(
        target_metadata=metadata,
        new_memory_usage=80.0,
        stage="interim",
    )

    assert result == {
        "old_memory_mb": 100.0,
        "new_memory_mb": 80.0,
        "change_mb": -20.0,
        "change_percentage": -20.0,
    }


def test_compute_memory_change_for_processed_stage_uses_nested_memory_usage() -> None:
    """Read baseline memory from processed metadata nested memory block."""
    metadata = {"memory": {"new_memory_mb": 50.0}}

    result = compute_memory_change(
        target_metadata=metadata,
        new_memory_usage=75.0,
        stage="processed",
    )

    assert result == {
        "old_memory_mb": 50.0,
        "new_memory_mb": 75.0,
        "change_mb": 25.0,
        "change_percentage": 50.0,
    }


def test_compute_memory_change_returns_zero_percentage_when_old_memory_is_zero() -> None:
    """Avoid division-by-zero and emit zero percentage when baseline is zero."""
    metadata = {"memory_usage_mb": 0.0}

    result = compute_memory_change(
        target_metadata=metadata,
        new_memory_usage=5.0,
        stage="interim",
    )

    assert result["change_percentage"] == 0


def test_compute_memory_change_raises_data_error_when_required_keys_missing() -> None:
    """Raise DataError when stage-specific baseline metadata key is absent."""
    metadata = {"memory": {}}

    with pytest.raises(DataError, match="missing the key required"):
        compute_memory_change(
            target_metadata=metadata,
            new_memory_usage=10.0,
            stage="processed",
        )
