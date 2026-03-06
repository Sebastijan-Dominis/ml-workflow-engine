"""Unit tests for 1D prediction-array normalization."""
import numpy as np
import pytest
from ml.exceptions import PipelineContractError
from ml.runners.shared.formatting.ensure_1d_array import ensure_1d_array

pytestmark = pytest.mark.unit


def test_ensure_1d_array_returns_numpy_array_for_valid_1d_input() -> None:
    """Verify conversion of valid 1D inputs to NumPy arrays."""
    result = ensure_1d_array([0.1, 0.2, 0.3])

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_ensure_1d_array_rejects_tuple_predictions() -> None:
    """Verify rejection of tuple-based prediction payloads."""
    with pytest.raises(PipelineContractError, match="Tuple predictions are not supported"):
        ensure_1d_array((np.array([1, 2]),))


def test_ensure_1d_array_rejects_non_1d_array() -> None:
    """Verify rejection of non-1D prediction payloads."""
    with pytest.raises(PipelineContractError, match="Expected 1D array of predictions"):
        ensure_1d_array([[1, 2], [3, 4]])


def test_ensure_1d_array_rejects_scalar_predictions() -> None:
    """Reject scalar outputs because they become 0D arrays instead of 1D vectors."""
    with pytest.raises(PipelineContractError, match="Expected 1D array of predictions"):
        ensure_1d_array(0.5)


def test_ensure_1d_array_accepts_empty_1d_inputs() -> None:
    """Accept empty sequences that still satisfy the 1D shape contract."""
    result = ensure_1d_array([])

    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)


def test_ensure_1d_array_normalizes_mixed_values_to_1d_array() -> None:
    """Normalize mixed-value lists into a valid 1D array without raising."""
    result = ensure_1d_array([1, "two", 3.0])

    assert result.shape == (3,)
    assert result.tolist() == ["1", "two", "3.0"]
