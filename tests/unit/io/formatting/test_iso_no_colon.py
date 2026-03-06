"""Unit tests for ISO timestamp formatting helper."""

from datetime import datetime

import pytest
from ml.io.formatting.iso_no_colon import iso_no_colon

pytestmark = pytest.mark.unit


def test_iso_no_colon_replaces_colons_in_time_component() -> None:
    """Format timestamps with hyphens instead of colons in the time component."""
    dt = datetime(2026, 3, 5, 14, 23, 59)

    result = iso_no_colon(dt)

    assert result == "2026-03-05T14-23-59"


def test_iso_no_colon_uses_seconds_precision() -> None:
    """Format timestamps at seconds precision (drop microseconds)."""
    dt = datetime(2026, 3, 5, 14, 23, 59, 987654)

    result = iso_no_colon(dt)

    assert result == "2026-03-05T14-23-59"
