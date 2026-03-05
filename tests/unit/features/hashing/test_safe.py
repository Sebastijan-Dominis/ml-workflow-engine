"""Unit tests for safe value-to-string conversion helper."""

import pytest
from ml.features.hashing.safe import safe

pytestmark = pytest.mark.unit


def test_safe_returns_literal_none_string_for_none_input() -> None:
    """Normalize None to a stable sentinel string for hash serialization."""
    assert safe(None) == "None"


def test_safe_delegates_to_str_for_non_none_values() -> None:
    """Stringify non-None values exactly using Python's str conversion."""
    assert safe(123) == "123"
    assert safe(True) == "True"
    assert safe("abc") == "abc"
