"""Unit tests for hotel-bookings row-id generation and lineage safeguards."""

from __future__ import annotations

import pandas as pd
import pytest
from ml.data.processed.processing.hotel_bookings.add_row_id import (
    AddRowIDToHotelBookings,
    validate_cols_for_row_id,
)
from ml.data.processed.processing.hotel_bookings.cols_for_row_id import cols_for_row_id
from ml.exceptions import DataError, UserError

pytestmark = pytest.mark.unit


def _row_template() -> dict[str, object]:
    """Return a valid minimal row containing every required row-id source column."""
    return {
        "hotel": "Resort Hotel",
        "arrival_date_year": 2017,
        "arrival_date_month": "July",
        "arrival_date_day_of_month": 1,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "country": "PRT",
        "agent": "9",
    }


def test_add_row_id_assigns_unique_ids_even_when_key_columns_contain_duplicates() -> None:
    """Guarantee uniqueness by suffixing duplicate-key rows with deterministic counters."""
    base = _row_template()
    duplicate_a = dict(base)
    duplicate_b = dict(base)
    variant = dict(base, assigned_room_type="B")
    source = pd.DataFrame([variant, duplicate_a, duplicate_b])

    result, metadata = AddRowIDToHotelBookings().add_row_id(source)

    assert len(result) == 3
    assert result["row_id"].is_unique

    row_id_parts = result["row_id"].str.rsplit("-", n=1, expand=True)
    prefixes = row_id_parts[0]
    suffixes = row_id_parts[1].astype(int)

    duplicate_prefixes = prefixes[prefixes.duplicated(keep=False)]
    assert duplicate_prefixes.nunique() == 1
    duplicate_suffixes = suffixes.loc[duplicate_prefixes.index].sort_values().tolist()
    assert duplicate_suffixes == [0, 1]

    assert metadata["cols_for_row_id"] == cols_for_row_id
    assert metadata["fingerprint"] == validate_cols_for_row_id()


def test_add_row_id_raises_data_error_when_any_required_column_is_missing() -> None:
    """Fail with a user-facing data error that identifies missing row-id source columns."""
    source = pd.DataFrame([_row_template()]).drop(columns=["agent"])

    with pytest.raises(DataError, match=r"required column\(s\) \['agent'\] are missing"):
        AddRowIDToHotelBookings().add_row_id(source)


def test_validate_cols_for_row_id_raises_user_error_on_fingerprint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block row-id generation when immutable column-fingerprint contract changes unexpectedly."""
    monkeypatch.setattr(
        "ml.data.processed.processing.hotel_bookings.add_row_id.COLS_FOR_ROW_ID_FINGERPRINT",
        "unexpected-fingerprint",
    )

    with pytest.raises(UserError, match="Cols for row_id have changed"):
        validate_cols_for_row_id()
