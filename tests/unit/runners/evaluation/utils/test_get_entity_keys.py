"""Unit tests for row identifier extraction helper."""

from __future__ import annotations

import pandas as pd
import pytest
from ml.exceptions import DataError
from ml.runners.evaluation.utils.get_entity_keys import get_entity_keys

pytestmark = pytest.mark.unit


def test_get_entity_keys_returns_original_row_id_series_object() -> None:
    """Return the exact row_id series object from the input dataframe."""
    df = pd.DataFrame({"row_id": [101, 102, 103], "feature": [1.2, 3.4, 5.6]})

    entity_keys = get_entity_keys(df, "row_id")

    assert entity_keys is df["row_id"]
    assert entity_keys.tolist() == [101, 102, 103]


def test_get_entity_keys_raises_data_error_when_row_id_is_missing() -> None:
    """Raise DataError with actionable guidance when row_id column is absent."""
    df = pd.DataFrame({"booking_id": [1, 2, 3]})

    with pytest.raises(
        DataError,
        match="does not contain a 'row_id' column",
    ):
        get_entity_keys(df, "row_id")


def test_get_entity_keys_logs_error_when_row_id_is_missing(caplog: pytest.LogCaptureFixture) -> None:
    """Emit error log for missing row_id before raising DataError."""
    df = pd.DataFrame({"booking_id": [1]})

    with caplog.at_level("ERROR", logger="ml.runners.evaluation.utils.get_entity_keys"), pytest.raises(
        DataError
    ):
        get_entity_keys(df, "row_id")

    assert "The data does not contain a 'row_id' column" in caplog.text
