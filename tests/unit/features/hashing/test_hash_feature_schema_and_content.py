"""Unit tests for feature schema and dataframe content hashing helpers."""

import pandas as pd
import pytest
from ml.features.hashing.hash_dataframe_content import hash_dataframe_content
from ml.features.hashing.hash_feature_schema import hash_feature_schema

pytestmark = pytest.mark.unit


def test_hash_feature_schema_is_stable_for_identical_schema() -> None:
    """Return same schema hash for dataframes with matching columns and dtypes."""
    df_a = pd.DataFrame({"a": pd.Series([1, 2], dtype="int64"), "b": pd.Series([1.0, 2.0], dtype="float64")})
    df_b = pd.DataFrame({"a": pd.Series([99, 100], dtype="int64"), "b": pd.Series([3.3, 4.4], dtype="float64")})

    assert hash_feature_schema(df_a) == hash_feature_schema(df_b)


def test_hash_feature_schema_changes_when_column_order_changes() -> None:
    """Use order-sensitive schema hashing to detect column reordering."""
    df_a = pd.DataFrame({"a": [1], "b": [2]})
    df_b = pd.DataFrame({"b": [2], "a": [1]})

    assert hash_feature_schema(df_a) != hash_feature_schema(df_b)


def test_hash_feature_schema_changes_when_dtype_changes() -> None:
    """Detect schema drift when a feature dtype changes."""
    df_int = pd.DataFrame({"a": pd.Series([1, 2], dtype="int64")})
    df_float = pd.DataFrame({"a": pd.Series([1.0, 2.0], dtype="float64")})

    assert hash_feature_schema(df_int) != hash_feature_schema(df_float)


def test_hash_dataframe_content_ignores_index_values() -> None:
    """Produce same content hash for identical row values with different indexes."""
    df_a = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=[10, 20])
    df_b = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=[100, 200])

    assert hash_dataframe_content(df_a) == hash_dataframe_content(df_b)


def test_hash_dataframe_content_changes_when_any_cell_value_changes() -> None:
    """Detect data content drift when at least one cell value differs."""
    df_a = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df_b = pd.DataFrame({"a": [1, 999], "b": ["x", "y"]})

    assert hash_dataframe_content(df_a) != hash_dataframe_content(df_b)
