"""Unit tests for row-id column fingerprint computation."""

import pytest
from ml.data.processed.processing.hotel_bookings.compute_cols_for_row_id_fingerprint import (
    compute_cols_for_row_id_fingerprint,
)

pytestmark = pytest.mark.unit


def test_compute_cols_for_row_id_fingerprint_is_order_insensitive() -> None:
    """Return identical fingerprints for equivalent column sets in different orders."""
    cols_a = ["hotel", "arrival_date", "booking_id"]
    cols_b = ["booking_id", "hotel", "arrival_date"]

    assert compute_cols_for_row_id_fingerprint(cols_a) == compute_cols_for_row_id_fingerprint(cols_b)


def test_compute_cols_for_row_id_fingerprint_changes_when_column_set_changes() -> None:
    """Return different fingerprints when source column membership changes."""
    cols_base = ["hotel", "arrival_date", "booking_id"]
    cols_extended = ["hotel", "arrival_date", "booking_id", "country"]

    assert compute_cols_for_row_id_fingerprint(cols_base) != compute_cols_for_row_id_fingerprint(cols_extended)
