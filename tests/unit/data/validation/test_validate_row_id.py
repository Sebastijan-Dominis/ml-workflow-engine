"""Unit tests for row identifier integrity validation."""

import pandas as pd
import pytest
from ml.data.validation.validate_row_id import validate_row_id
from ml.exceptions import DataError

pytestmark = pytest.mark.unit


def test_validate_row_id_passes_when_column_exists_and_values_are_unique() -> None:
    """Accept feature sets with present and unique row identifiers."""
    df = pd.DataFrame({"row_id": [1, 2, 3], "feature_a": [10, 20, 30]})

    validate_row_id(df)


def test_validate_row_id_raises_when_row_id_column_is_missing() -> None:
    """Reject feature sets that do not include the required `row_id` column."""
    df = pd.DataFrame({"feature_a": [10, 20, 30]})

    with pytest.raises(DataError, match="must contain a 'row_id' column"):
        validate_row_id(df)


def test_validate_row_id_raises_with_duplicate_values_listed() -> None:
    """Reject feature sets with duplicate row IDs and expose offending values."""
    df = pd.DataFrame({"row_id": [1, 2, 2, 3], "feature_a": [10, 20, 30, 40]})

    with pytest.raises(DataError, match=r"Found duplicates: \[2\]"):
        validate_row_id(df)
