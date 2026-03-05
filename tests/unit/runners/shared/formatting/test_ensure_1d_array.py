import numpy as np
import pytest
from ml.exceptions import PipelineContractError
from ml.runners.shared.formatting.ensure_1d_array import ensure_1d_array

pytestmark = pytest.mark.unit


def test_ensure_1d_array_returns_numpy_array_for_valid_1d_input() -> None:
    result = ensure_1d_array([0.1, 0.2, 0.3])

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_ensure_1d_array_rejects_tuple_predictions() -> None:
    with pytest.raises(PipelineContractError, match="Tuple predictions are not supported"):
        ensure_1d_array((np.array([1, 2]),))


def test_ensure_1d_array_rejects_non_1d_array() -> None:
    with pytest.raises(PipelineContractError, match="Expected 1D array of predictions"):
        ensure_1d_array([[1, 2], [3, 4]])
