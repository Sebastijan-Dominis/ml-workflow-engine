"""Unit tests for deterministic metadata hashing using real Arrow/Parquet files."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest
from ml.features.hashing.hash_arrow_metadata import hash_arrow_metadata
from ml.features.hashing.hash_parquet_metadata import hash_parquet_metadata

pytestmark = pytest.mark.unit


def test_hash_arrow_metadata_is_deterministic_for_same_file(tmp_path: Path) -> None:
    """Return identical hash values when hashing the same Arrow file repeatedly."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    table = pa.Table.from_pandas(df)
    path = tmp_path / "features.arrow"
    feather.write_feather(table, path)

    assert hash_arrow_metadata(path) == hash_arrow_metadata(path)


def test_hash_parquet_metadata_is_deterministic_for_same_file(tmp_path: Path) -> None:
    """Return identical hash values when hashing the same Parquet file repeatedly."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    table = pa.Table.from_pandas(df)
    path = tmp_path / "features.parquet"
    pq.write_table(table, path)

    assert hash_parquet_metadata(path) == hash_parquet_metadata(path)


def test_hash_parquet_metadata_changes_when_column_schema_changes(tmp_path: Path) -> None:
    """Detect schema-level drift between different Parquet snapshots."""
    df_a = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df_b = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})

    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    pq.write_table(pa.Table.from_pandas(df_a), path_a)
    pq.write_table(pa.Table.from_pandas(df_b), path_b)

    assert hash_parquet_metadata(path_a) != hash_parquet_metadata(path_b)
