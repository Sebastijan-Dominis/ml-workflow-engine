"""A module for hashing input rows in the inference pipeline."""
import hashlib

import pandas as pd


def hash_input_row(row: pd.Series) -> str:
    """Generate a stable hash for a given input row.

    Args:
        row: A pandas Series representing a single input row.

    Returns:
        A hexadecimal string representing the hash of the input row."""
    row = row.sort_index()

    normalized = []
    for v in row.values:
        if pd.isna(v):
            normalized.append("NULL")
        elif isinstance(v, float):
            normalized.append(f"{v:.10g}")  # stable float format
        else:
            normalized.append(str(v))

    row_str = "|".join(normalized)
    return hashlib.sha256(row_str.encode()).hexdigest()
