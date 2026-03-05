"""Unit tests for the ensure_1d_array function in ml.runners.shared.formatting.ensure_1d_array. The tests verify that ensure_1d_array correctly converts valid 1D input into a numpy array, and raises PipelineContractError when given invalid input such as tuples or non-1D arrays."""
import numpy as np
import pytest
from ml.exceptions import PipelineContractError
from ml.runners.shared.formatting.ensure_1d_array import ensure_1d_array

pytestmark = pytest.mark.unit


def test_ensure_1d_array_returns_numpy_array_for_valid_1d_input() -> None:
    """Test that ensure_1d_array correctly converts a valid 1D list of predictions into a numpy array with the expected shape."""
    result = ensure_1d_array([0.1, 0.2, 0.3])

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_ensure_1d_array_rejects_tuple_predictions() -> None:
    """Test that ensure_1d_array raises a PipelineContractError when a tuple of predictions is provided instead of a 1D array or list."""
    with pytest.raises(PipelineContractError, match="Tuple predictions are not supported"):
        ensure_1d_array((np.array([1, 2]),))


def test_ensure_1d_array_rejects_non_1d_array() -> None:
    """Test that ensure_1d_array raises a PipelineContractError when a non-1D array (e.g., 2D array) is provided."""
    with pytest.raises(PipelineContractError, match="Expected 1D array of predictions"):
        ensure_1d_array([[1, 2], [3, 4]])
