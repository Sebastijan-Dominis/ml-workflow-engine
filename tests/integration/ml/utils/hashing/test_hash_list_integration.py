"""Integration tests for list hashing utilities."""

from __future__ import annotations

from ml.utils.hashing.hash_list import hash_list


def test_hash_list_order_matters_and_not() -> None:
    a = [1, 2, 3]
    b = [3, 2, 1]

    # When order matters the hashes should differ
    assert hash_list(a, order_matters=True) != hash_list(b, order_matters=True)

    # When order does not matter the hashes should be equal
    assert hash_list(a, order_matters=False) == hash_list(b, order_matters=False)
