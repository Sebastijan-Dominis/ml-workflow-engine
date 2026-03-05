"""Unit tests for dataframe memory usage computation helper."""

import pandas as pd
import pytest
from ml.data.utils.memory.get_memory_usage import get_memory_usage
from ml.exceptions import RuntimeMLError

pytestmark = pytest.mark.unit


def test_get_memory_usage_returns_positive_float_for_valid_dataframe() -> None:
    """Return memory usage in MB as a positive floating-point number."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "yy", "zzz"]})

    result = get_memory_usage(df)

    assert isinstance(result, float)
    assert result > 0


def test_get_memory_usage_wraps_internal_dataframe_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap unexpected dataframe memory errors into RuntimeMLError."""
    df = pd.DataFrame({"a": [1]})

    def _raise(*args: object, **kwargs: object) -> object:
        raise ValueError("boom")

    monkeypatch.setattr(df, "memory_usage", _raise)

    with pytest.raises(RuntimeMLError, match="Error computing memory usage"):
        get_memory_usage(df)
