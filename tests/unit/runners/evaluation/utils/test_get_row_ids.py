"""Unit tests for row identifier extraction helper."""

from __future__ import annotations

import pandas as pd
import pytest
from ml.exceptions import DataError
from ml.runners.evaluation.utils.get_row_ids import get_row_ids

pytestmark = pytest.mark.unit


def test_get_row_ids_returns_original_row_id_series_object() -> None:
    """Return the exact row_id series object from the input dataframe."""
    df = pd.DataFrame({"row_id": [101, 102, 103], "feature": [1.2, 3.4, 5.6]})

    row_ids = get_row_ids(df)

    assert row_ids is df["row_id"]
    assert row_ids.tolist() == [101, 102, 103]


def test_get_row_ids_raises_data_error_when_row_id_is_missing() -> None:
    """Raise DataError with actionable guidance when row_id column is absent."""
    df = pd.DataFrame({"booking_id": [1, 2, 3]})

    with pytest.raises(
        DataError,
        match="does not contain a 'row_id' column",
    ):
        get_row_ids(df)


def test_get_row_ids_logs_error_when_row_id_is_missing(caplog: pytest.LogCaptureFixture) -> None:
    """Emit error log for missing row_id before raising DataError."""
    df = pd.DataFrame({"booking_id": [1]})

    with caplog.at_level("ERROR", logger="ml.runners.evaluation.utils.get_row_ids"), pytest.raises(
        DataError
    ):
        get_row_ids(df)

    assert "The data does not contain a 'row_id' column" in caplog.text
