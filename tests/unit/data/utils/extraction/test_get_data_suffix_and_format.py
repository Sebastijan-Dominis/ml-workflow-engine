"""Unit tests for metadata extraction of data path suffix and format."""

import pytest
from ml.data.utils.extraction.get_data_suffix_and_format import (
    get_data_suffix_and_format,
)
from ml.exceptions import UserError

pytestmark = pytest.mark.unit


def test_get_data_suffix_and_format_reads_data_location_fields() -> None:
    """Resolve path suffix and format from top-level `data` metadata location."""
    metadata = {"data": {"path_suffix": "hotel_bookings", "format": "parquet"}}

    suffix, data_format = get_data_suffix_and_format(metadata, "data")

    assert suffix == "hotel_bookings"
    assert data_format == "parquet"


def test_get_data_suffix_and_format_reads_data_output_location_fields() -> None:
    """Resolve path suffix and format from nested `data.output` metadata location."""
    metadata = {
        "data": {
            "output": {
                "path_suffix": "processed/hotel_bookings",
                "format": "parquet",
            }
        }
    }

    suffix, data_format = get_data_suffix_and_format(metadata, "data/output")

    assert suffix == "processed/hotel_bookings"
    assert data_format == "parquet"


def test_get_data_suffix_and_format_raises_for_invalid_location() -> None:
    """Reject metadata location values outside supported extraction paths."""
    metadata = {"data": {"path_suffix": "x", "format": "csv"}}

    with pytest.raises(UserError, match="Invalid location"):
        get_data_suffix_and_format(metadata, "invalid")  # type: ignore[arg-type]


def test_get_data_suffix_and_format_raises_when_path_suffix_missing() -> None:
    """Require data path suffix to build data file path correctly."""
    metadata = {"data": {"format": "csv"}}

    with pytest.raises(UserError, match="missing 'data.path_suffix'"):
        get_data_suffix_and_format(metadata, "data")


def test_get_data_suffix_and_format_raises_when_format_missing() -> None:
    """Require data format metadata needed for loader selection."""
    metadata = {"data": {"path_suffix": "hotel_bookings"}}

    with pytest.raises(UserError, match="missing 'data.format'"):
        get_data_suffix_and_format(metadata, "data")
