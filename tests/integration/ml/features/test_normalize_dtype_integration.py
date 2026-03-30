"""Integration tests for dtype normalization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ml.features.validation.normalize_dtype import normalize_dtype


def test_normalize_common_dtypes() -> None:
    assert normalize_dtype(np.dtype("int64")) == "int64"
    assert normalize_dtype(np.dtype("float64")) == "float64"
    assert normalize_dtype(np.dtype("bool")) == "bool"
    assert normalize_dtype(np.dtype("datetime64[ns]")) == "datetime64[ns]"


def test_normalize_pandas_nullable_and_category() -> None:
    s_str = pd.Series(["a"], dtype="string")
    assert normalize_dtype(s_str.dtype) == "object"

    s_cat = pd.Series(["x", "y"], dtype="category")
    assert normalize_dtype(s_cat.dtype) == "category"

    s_int = pd.Series([1, None], dtype="Int64")
    assert normalize_dtype(s_int.dtype) == "int64"
