"""Hashing helpers for in-memory dataframe content fingerprints."""

import hashlib

import pandas as pd


def hash_dataframe_content(X: pd.DataFrame) -> str:
    """Return an MD5 hash of dataframe row content excluding index values.

    Args:
        X: Dataframe whose row content should be hashed.

    Returns:
        Content hash string for the dataframe rows.
    """

    arr = pd.util.hash_pandas_object(X, index=False).to_numpy()
    return hashlib.md5(arr.tobytes()).hexdigest()
