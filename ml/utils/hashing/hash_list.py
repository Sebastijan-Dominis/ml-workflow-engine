"""Hashing helpers for list payloads with optional order invariance."""

import hashlib
import json

def hash_list(lst: list, order_matters: bool = True) -> str:
    """Return a SHA-256 hash for a JSON-normalized list representation.

    Args:
        lst: List payload to hash.
        order_matters: Whether list order should affect the resulting hash.

    Returns:
        SHA-256 hash of the normalized list representation.
    """

    if not order_matters:
        lst = sorted(lst)
    normalized = json.dumps(lst, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest()