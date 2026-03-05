"""Unit tests for Arrow and Parquet metadata hashing wrapper behavior."""

from pathlib import Path

import pytest
from ml.exceptions import RuntimeMLError
from ml.features.hashing.hash_arrow_metadata import hash_arrow_metadata
from ml.features.hashing.hash_parquet_metadata import hash_parquet_metadata

pytestmark = pytest.mark.unit


def test_hash_parquet_metadata_wraps_underlying_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise RuntimeMLError when parquet metadata reading or hashing fails."""

    def _raise(*args: object, **kwargs: object) -> object:
        raise ValueError("boom")

    monkeypatch.setattr("ml.features.hashing.hash_parquet_metadata.pq.ParquetFile", _raise)

    with pytest.raises(RuntimeMLError, match="Error hashing Parquet metadata"):
        hash_parquet_metadata(Path("dummy.parquet"))


def test_hash_arrow_metadata_wraps_underlying_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise RuntimeMLError when Arrow memory map or metadata reading fails."""

    def _raise(*args: object, **kwargs: object) -> object:
        raise ValueError("boom")

    monkeypatch.setattr("ml.features.hashing.hash_arrow_metadata.pa.memory_map", _raise)

    with pytest.raises(RuntimeMLError, match="Failed to hash Arrow metadata"):
        hash_arrow_metadata(Path("dummy.arrow"))
