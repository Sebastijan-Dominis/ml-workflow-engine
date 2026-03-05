"""Unit tests for list hashing utility with optional order sensitivity."""

import pytest
from ml.utils.hashing.hash_list import hash_list

pytestmark = pytest.mark.unit


def test_hash_list_is_order_sensitive_by_default() -> None:
    """Produce different hashes when list ordering differs and order matters."""
    assert hash_list(["a", "b"]) != hash_list(["b", "a"])


def test_hash_list_is_order_insensitive_when_configured() -> None:
    """Produce same hash for equivalent lists when order sensitivity is disabled."""
    assert hash_list(["a", "b"], order_matters=False) == hash_list(["b", "a"], order_matters=False)


def test_hash_list_changes_when_membership_changes() -> None:
    """Detect content drift when list members differ despite same size."""
    assert hash_list(["a", "b"], order_matters=False) != hash_list(["a", "c"], order_matters=False)


def test_hash_list_supports_special_ascii_punctuation_values_deterministically() -> None:
    """Hash punctuation-rich ASCII values consistently under order-insensitive mode."""
    assert hash_list(["a-b", "x_y"], order_matters=False) == hash_list(["x_y", "a-b"], order_matters=False)
