"""Unit tests for processed-stage dataframe transformation helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from ml.data.config.schemas.processed import ProcessedConfig
from ml.data.processed.processing.process_data import add_row_id, remove_columns
from ml.exceptions import DataError, UserError

pytestmark = pytest.mark.unit


def test_remove_columns_drops_requested_columns_and_preserves_remaining_order() -> None:
    """Drop configured columns while preserving unremoved column order and values."""
    df = pd.DataFrame(
        {
            "booking_id": [1, 2],
            "noise_a": [10, 20],
            "feature_x": [0.5, 0.7],
            "noise_b": [100, 200],
        }
    )

    out = remove_columns(df, ["noise_a", "noise_b"])

    assert out.columns.tolist() == ["booking_id", "feature_x"]
    assert out["booking_id"].tolist() == [1, 2]
    assert out["feature_x"].tolist() == [0.5, 0.7]


def test_remove_columns_raises_data_error_with_missing_column_names() -> None:
    """Reject removal requests containing any columns that are not present in the frame."""
    df = pd.DataFrame({"booking_id": [1], "feature_x": [0.5]})

    with pytest.raises(DataError, match=r"Cannot remove columns \['missing_col'\]"):
        remove_columns(df, ["missing_col"])


def test_remove_columns_with_empty_request_returns_equivalent_dataframe() -> None:
    """Return an unchanged-equivalent frame when no columns are requested for removal."""
    df = pd.DataFrame({"booking_id": [1, 2], "feature_x": [0.1, 0.2]})

    out = remove_columns(df, [])

    assert out.equals(df)


def test_add_row_id_dispatches_to_registered_row_id_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call the row-id generator registered for the dataset and return its outputs."""
    df = pd.DataFrame({"booking_id": [1, 2], "feature_x": [5.0, 6.0]})
    cfg = cast(ProcessedConfig, SimpleNamespace(data=SimpleNamespace(name="hotel_bookings")))

    expected_df = df.assign(row_id=["r1", "r2"])
    expected_meta = SimpleNamespace(method="fingerprint", columns=["booking_id", "feature_x"])

    monkeypatch.setattr(
        "ml.data.processed.processing.process_data.ROW_ID_FUNCTIONS",
        {
            "hotel_bookings": lambda frame: (expected_df, expected_meta),
        },
    )

    out_df, out_meta = add_row_id(df, cfg)

    assert out_df.equals(expected_df)
    assert out_meta is expected_meta


def test_add_row_id_raises_user_error_when_dataset_has_no_registered_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a clear user-facing error when row-id generation policy is missing."""
    df = pd.DataFrame({"booking_id": [1]})
    cfg = cast(ProcessedConfig, SimpleNamespace(data=SimpleNamespace(name="unknown_dataset")))

    monkeypatch.setattr("ml.data.processed.processing.process_data.ROW_ID_FUNCTIONS", {})

    with pytest.raises(UserError, match="Row ID generation is not implemented"):
        add_row_id(df, cfg)
