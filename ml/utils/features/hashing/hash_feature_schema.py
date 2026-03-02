"""Hashing helpers for feature schema structure fingerprints."""

import hashlib

import pandas as pd


def hash_feature_schema(X: pd.DataFrame) -> str:
    """Compute a stable hash from feature names and dtype signatures.

    Args:
        X: Dataframe whose schema should be hashed.

    Returns:
        Stable schema hash string.
    """

    h = hashlib.sha256()
    for col in X.columns:
        h.update(col.encode())
        h.update(str(X[col].dtype).encode())
    return h.hexdigest()