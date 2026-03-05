"""Unit tests for ISO timestamp formatting helper."""

from datetime import datetime

import pytest
from ml.io.formatting.iso_no_colon import iso_no_colon

pytestmark = pytest.mark.unit


def test_iso_no_colon_replaces_colons_in_time_component() -> None:
    """Test that iso_no_colon correctly formats a datetime object into an ISO string without colons in the time component. The test creates a datetime object with a specific date and time, calls iso_no_colon with this datetime, and asserts that the returned string matches the expected ISO format where the time component uses hyphens instead of colons, confirming that iso_no_colon correctly formats the datetime as intended."""
    dt = datetime(2026, 3, 5, 14, 23, 59)

    result = iso_no_colon(dt)

    assert result == "2026-03-05T14-23-59"


def test_iso_no_colon_uses_seconds_precision() -> None:
    """Test that iso_no_colon includes seconds precision in the formatted timestamp. The test creates a datetime object with microseconds, calls iso_no_colon with this datetime, and asserts that the returned string includes the seconds component, confirming that iso_no_colon correctly handles seconds precision."""
    dt = datetime(2026, 3, 5, 14, 23, 59, 987654)

    result = iso_no_colon(dt)

    assert result == "2026-03-05T14-23-59"
