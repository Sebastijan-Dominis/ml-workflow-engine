"""Unit tests for canonical dtype normalization helpers."""

import numpy as np
import pandas as pd
import pytest
from ml.features.validation.normalize_dtype import normalize_dtype

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pd.CategoricalDtype(categories=["a", "b"], ordered=False), "category"),
        (pd.StringDtype(storage="python"), "object"),
        (np.dtype("int64"), "int64"),
        (np.dtype("float64"), "float64"),
        (np.dtype("bool"), "bool"),
        (np.dtype("O"), "object"),
        (np.dtype("datetime64[ns]"), "datetime64[ns]"),
    ],
)
def test_normalize_dtype_maps_common_dtypes_to_canonical_labels(
    dtype: object,
    expected: str,
) -> None:
    """Map common pandas/NumPy dtypes to stable labels used by validation logic."""
    assert normalize_dtype(dtype) == expected


@pytest.mark.parametrize("dtype", ["Int64", "Int32", "UInt8", "UInt64"])
def test_normalize_dtype_maps_nullable_integer_strings_to_int64(dtype: str) -> None:
    """Normalize nullable integer extension dtypes to canonical `int64` label."""
    assert normalize_dtype(dtype) == "int64"


def test_normalize_dtype_maps_nullable_integer_extension_dtype_to_int64() -> None:
    """Handle pandas nullable integer extension dtype objects directly."""
    series = pd.Series([1, None], dtype="Int64")

    assert normalize_dtype(series.dtype) == "int64"


def test_normalize_dtype_returns_raw_string_for_unhandled_custom_dtype() -> None:
    """Fallback to string form for dtype-like values outside handled categories."""

    class _CustomDType:
        def __str__(self) -> str:
            return "my_custom_dtype"

    assert normalize_dtype(_CustomDType()) == "my_custom_dtype"
