"""Unit tests for feature data-loading and lineage-validation helpers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from ml.exceptions import ConfigError, DataError
from ml.types import DataLineageEntry

pytestmark = pytest.mark.unit

# `ml.types` imports CatBoost classes at module import time; provide a lightweight stub.
if "catboost" not in sys.modules:
    catboost_stub = cast(Any, types.ModuleType("catboost"))
    catboost_stub.CatBoostClassifier = type("CatBoostClassifier", (), {})
    catboost_stub.CatBoostRegressor = type("CatBoostRegressor", (), {})
    sys.modules["catboost"] = catboost_stub

_data_loader_module = importlib.import_module("ml.features.loading.data_loader")
lineage_identity = _data_loader_module.lineage_identity
load_and_validate_data = _data_loader_module.load_and_validate_data


def _entry(
    *,
    path: Path,
    fmt: str,
    merge_key: str = "row_id",
    loader_hash: str = "loader-hash",
    data_hash: str = "data-hash",
) -> DataLineageEntry:
    """Build a minimal lineage entry accepted by the data-loading helper."""
    return DataLineageEntry(
        ref="feature_store",
        name="booking_context_features",
        version="v1",
        format=fmt,
        path_suffix=f"features.{fmt}",
        merge_key=merge_key,
        snapshot_id="snapshot-1",
        path=str(path),
        loader_validation_hash=loader_hash,
        data_hash=data_hash,
        row_count=2,
        column_count=2,
    )


def test_lineage_identity_returns_expected_comparison_tuple() -> None:
    """Project each entry into the identity fields used by lineage reconciliation checks."""
    entry = _entry(path=Path("/tmp/features.parquet"), fmt="parquet")

    identity = lineage_identity(entry)

    assert identity == (
        "booking_context_features",
        "v1",
        "snapshot-1",
        "data-hash",
        "loader-hash",
        "row_id",
    )


def test_load_and_validate_data_raises_when_no_lineage_entries() -> None:
    """Reject empty lineage because no datasets can be loaded or merged."""
    with pytest.raises(ConfigError, match="No datasets specified"):
        load_and_validate_data([])


def test_load_and_validate_data_raises_for_unsupported_format(tmp_path: Path) -> None:
    """Reject entries whose format has no registered loader-hash implementation."""
    dataset_entry = _entry(path=tmp_path / "features.unknown", fmt="unknown")

    with pytest.raises(ConfigError, match="Unsupported data format"):
        load_and_validate_data([dataset_entry])


def test_load_and_validate_data_raises_when_dataset_file_does_not_exist(tmp_path: Path) -> None:
    """Raise ``DataError`` when a lineage entry points to a missing dataset file."""
    fmt = "csv"
    missing_path = tmp_path / "missing.csv"
    dataset_entry = _entry(path=missing_path, fmt=fmt)

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(_data_loader_module.HASH_LOADER_REGISTRY, fmt, lambda _p: "loader-hash")

        with pytest.raises(DataError, match="Dataset file not found"):
            load_and_validate_data([dataset_entry])


def test_load_and_validate_data_returns_merged_dataframe_on_valid_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load and merge dataset entries, then return dataframe when lineage identities match."""
    fmt = "csv"
    dataset_path = tmp_path / "features.csv"
    dataset_path.write_text("placeholder", encoding="utf-8")

    dataset_entry = _entry(path=dataset_path, fmt=fmt, loader_hash="loader-ok", data_hash="data-ok")
    df = pd.DataFrame({"row_id": [1, 2], "feature": [0.1, 0.2]})

    monkeypatch.setattr("ml.features.loading.data_loader.read_data", lambda _fmt, _path: df)
    monkeypatch.setattr(
        "ml.features.loading.data_loader.merge_dataset_into_main",
        lambda **kwargs: (kwargs["df"], "data-ok"),
    )
    monkeypatch.setitem(_data_loader_module.HASH_LOADER_REGISTRY, fmt, lambda _p: "loader-ok")

    result = load_and_validate_data([dataset_entry])

    assert result.equals(df)


def test_load_and_validate_data_raises_when_actual_lineage_differs_from_expected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise ``DataError`` when recomputed lineage identity does not match input lineage."""
    fmt = "csv"
    dataset_path = tmp_path / "features.csv"
    dataset_path.write_text("placeholder", encoding="utf-8")

    dataset_entry = _entry(path=dataset_path, fmt=fmt, loader_hash="expected-loader", data_hash="expected-data")
    df = pd.DataFrame({"row_id": [1, 2], "feature": [0.1, 0.2]})

    monkeypatch.setattr("ml.features.loading.data_loader.read_data", lambda _fmt, _path: df)
    monkeypatch.setattr(
        "ml.features.loading.data_loader.merge_dataset_into_main",
        lambda **kwargs: (kwargs["df"], "actual-data"),
    )
    monkeypatch.setitem(_data_loader_module.HASH_LOADER_REGISTRY, fmt, lambda _p: "actual-loader")

    with pytest.raises(DataError, match="Data lineage mismatch"):
        load_and_validate_data([dataset_entry])
