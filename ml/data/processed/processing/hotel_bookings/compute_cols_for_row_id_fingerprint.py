"""Utilities for deriving stable fingerprints for row-id source columns."""

from ml.utils.hashing.hash_list import hash_list


def compute_cols_for_row_id_fingerprint(cols):
    """Compute an order-insensitive fingerprint for row-id column definitions.

    Args:
        cols: Iterable of column names used to derive row identifiers.

    Returns:
        Stable hash fingerprint for the provided column set.
    """

    return hash_list(cols, order_matters=False)