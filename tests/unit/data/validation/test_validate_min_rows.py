"""Unit tests for minimum-row validation safeguards."""

import pandas as pd
import pytest
from ml.data.validation.validate_min_rows import validate_min_rows
from ml.exceptions import DataError

pytestmark = pytest.mark.unit


def test_validate_min_rows_passes_when_row_count_equals_threshold() -> None:
    """Accept datasets whose row count is exactly the configured minimum."""
    df = pd.DataFrame({"x": [1, 2, 3]})

    validate_min_rows(df, min_rows=3)


def test_validate_min_rows_raises_when_row_count_is_below_threshold() -> None:
    """Reject datasets that do not satisfy configured minimum row count."""
    df = pd.DataFrame({"x": [1, 2]})

    with pytest.raises(DataError, match="less than the minimum required"):
        validate_min_rows(df, min_rows=3)


def test_validate_min_rows_logs_warning_and_defaults_to_zero_when_unset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn and skip strict row-count enforcement when minimum is not configured."""
    df = pd.DataFrame({"x": []})

    with caplog.at_level("WARNING"):
        validate_min_rows(df, min_rows=0)

    assert "Minimum rows constraint not set" in caplog.text
