"""Utility functions for feature drifting monitoring."""
from typing import Literal

import pandas as pd


def infer_drift_method(series: pd.Series) -> Literal["ks", "psi"]:
    """
    Decide which drift metric to use for a feature.

    Returns:
        "ks" or "psi"
    """
    if pd.api.types.is_numeric_dtype(series):
        # Low cardinality numeric behaves like categorical
        if series.nunique(dropna=True) < 20:
            return "psi"
        return "ks"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "psi"  # always bin → PSI

    # everything else → categorical
    return "psi"
