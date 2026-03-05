"""Utilities for normalizing pandas dtypes to canonical string labels."""

import numpy as np


def normalize_dtype(dtype) -> str:
    """Normalize pandas and extension dtypes to stable string categories.

    Args:
        dtype: Pandas or NumPy dtype object.

    Returns:
        Canonical dtype category string.
    """

    # Handle categorical
    if hasattr(dtype, "categories") and hasattr(dtype, "ordered"):
        return "category"

    # Handle nullable string dtype
    if str(dtype) == "string[python]" or str(dtype) == "string":
        return "object"

    # Handle nullable integers (Int64, Int32, Int16, Int8)
    if str(dtype).startswith("Int") or str(dtype).startswith("UInt"):
        return "int64"

    # Some custom dtype-like objects cannot be interpreted by np.issubdtype.
    # In that case we preserve the original fallback behavior and return str(dtype).
    try:
        if np.issubdtype(dtype, np.integer):
            return "int64"
        if np.issubdtype(dtype, np.floating):
            return "float64"
        if np.issubdtype(dtype, np.bool_):
            return "bool"
        if np.issubdtype(dtype, np.object_):
            return "object"
        if np.issubdtype(dtype, np.datetime64):
            return "datetime64[ns]"
    except TypeError:
        return str(dtype)

    return str(dtype)
