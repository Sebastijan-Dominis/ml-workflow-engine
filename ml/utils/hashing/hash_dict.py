"""Hashing helpers for deterministic dictionary fingerprints."""

import hashlib
import json


def canonicalize(obj):
    """Recursively canonicalize nested containers into stable ordering.

    Args:
        obj: Arbitrary nested structure.

    Returns:
        Any: Canonicalized structure.
    """

    if isinstance(obj, dict):
        return {k: canonicalize(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, list):
        return [canonicalize(x) for x in obj]
    elif isinstance(obj, set):
        return sorted(canonicalize(x) for x in obj)
    else:
        return obj

def hash_dict(d: dict) -> str:
    """Return a SHA-256 hash for a canonicalized dictionary payload.

    Args:
        d: Input dictionary payload.

    Returns:
        str: SHA-256 hash digest.
    """

    normalized = json.dumps(
        canonicalize(d),
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode()).hexdigest()
