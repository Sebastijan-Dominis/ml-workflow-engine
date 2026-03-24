"""Unit tests for row identifier integrity validation."""

import pandas as pd
import pytest
from ml.data.validation.validate_entity_key import validate_entity_key
from ml.exceptions import DataError

pytestmark = pytest.mark.unit


def test_validate_entity_key_passes_when_column_exists_and_values_are_unique() -> None:
    """Accept feature sets with present and unique row identifiers."""
    df = pd.DataFrame({"entity_key": [1, 2, 3], "feature_a": [10, 20, 30]})

    validate_entity_key(df, "entity_key")


def test_validate_entity_key_raises_when_row_id_column_is_missing() -> None:
    """Reject feature sets that do not include the required `entity_key` column."""
    df = pd.DataFrame({"feature_a": [10, 20, 30]})

    with pytest.raises(DataError, match="must contain a 'entity_key' column"):
        validate_entity_key(df, "entity_key")


def test_validate_entity_key_raises_with_duplicate_values_listed() -> None:
    """Reject feature sets with duplicate entity keys and expose offending values."""
    df = pd.DataFrame({"entity_key": [1, 2, 2, 3], "feature_a": [10, 20, 30, 40]})

    with pytest.raises(DataError, match=r"Found duplicates"):
        validate_entity_key(df, "entity_key")
