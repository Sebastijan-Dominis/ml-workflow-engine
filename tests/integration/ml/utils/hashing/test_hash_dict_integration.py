"""Integration tests for dictionary hashing utilities."""

from __future__ import annotations

from typing import Any, cast

from ml.utils.hashing.hash_dict import canonicalize, hash_dict


def test_canonicalize_and_hash_dict_order_invariance() -> None:
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 2, "a": 1}

    # canonicalize should produce the same logical structure irrespective
    # of insertion order
    assert canonicalize(d1) == canonicalize(d2)

    # hash should be identical for order-insensitive dictionaries
    assert hash_dict(d1) == hash_dict(d2)


def test_canonicalize_handles_nested_and_sets() -> None:
    payload = {"z": {2, 1}, "x": [3, {"b": 1, "a": 2}]}
    canon = cast(dict[str, Any], canonicalize(payload))

    # sets become sorted lists and nested dicts are ordered
    assert isinstance(canon["z"], list)
    assert canon["z"] == [1, 2]
    assert isinstance(canon["x"], list)
