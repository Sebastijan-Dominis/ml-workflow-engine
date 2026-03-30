"""Integration tests for small hashing helper `safe`."""

from __future__ import annotations

from ml.features.hashing.safe import safe


def test_safe_none_and_values() -> None:
    assert safe(None) == "None"
    assert safe(123) == "123"
    assert safe("abc") == "abc"
