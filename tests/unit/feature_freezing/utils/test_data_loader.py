"""Unit tests for dataset loading and lineage assembly helper."""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.exceptions import ConfigError, DataError

pytestmark = pytest.mark.unit

# ml.types imports catboost; stub it to keep unit tests dependency-light.
if "catboost" not in sys.modules:
    catboost_stub = cast(Any, types.ModuleType("catboost"))
    catboost_stub.CatBoostClassifier = type("CatBoostClassifier", (), {})
    catboost_stub.CatBoostRegressor = type("CatBoostRegressor", (), {})
    sys.modules["catboost"] = catboost_stub

_data_loader_module = importlib.import_module("ml.feature_freezing.utils.data_loader")
load_data_with_lineage = _data_loader_module.load_data_with_lineage


def _as_any_config(config: SimpleNamespace) -> Any:
    """Cast minimal config stubs for invoking the typed loader function."""
    return cast(Any, config)

def _dataset_stub(
    *,
    ref: str,
    name: str,
    version: str,
    fmt: str,
    path_suffix: str,
    merge_key: str,
) -> SimpleNamespace:
    """Build minimal dataset config-like object consumed by loader utility."""
    return SimpleNamespace(
        ref=ref,
        name=name,
        version=version,
        format=fmt,
        path_suffix=path_suffix,
        merge_key=merge_key,
    )


def test_load_data_with_lineage_raises_when_no_datasets_specified() -> None:
    """Reject empty data config because no datasets can be loaded or merged."""
    cfg = SimpleNamespace(data=[])

    with pytest.raises(ConfigError, match="No datasets specified"):
        load_data_with_lineage(_as_any_config(cfg), snapshot_binding_key=None)


def test_load_data_with_lineage_raises_for_unsupported_dataset_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject datasets whose format has no loader/hash implementation registered."""
    dataset = _dataset_stub(
        ref=str(tmp_path),
        name="booking_context_features",
        version="v1",
        fmt="unsupported_format",
        path_suffix="data.{format}",
        merge_key="row_id",
    )
    cfg = SimpleNamespace(data=[dataset])

    monkeypatch.setattr(
        "ml.feature_freezing.utils.data_loader.get_latest_snapshot_path",
        lambda path: tmp_path,
    )

    with pytest.raises(ConfigError, match="Unsupported data format"):
        load_data_with_lineage(_as_any_config(cfg), snapshot_binding_key=None)


def test_load_data_with_lineage_raises_when_dataset_file_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject datasets when resolved snapshot file path does not exist."""
    fmt = next(iter(_data_loader_module.HASH_LOADER_REGISTRY))
    dataset = _dataset_stub(
        ref=str(tmp_path),
        name="booking_context_features",
        version="v1",
        fmt=fmt,
        path_suffix="missing_file.{format}",
        merge_key="row_id",
    )
    cfg = SimpleNamespace(data=[dataset])

    monkeypatch.setattr(
        "ml.feature_freezing.utils.data_loader.get_latest_snapshot_path",
        lambda path: tmp_path,
    )

    with pytest.raises(DataError, match="Dataset file not found"):
        load_data_with_lineage(_as_any_config(cfg), snapshot_binding_key=None)


def test_load_data_with_lineage_returns_merged_data_and_lineage_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return merged dataframe and complete lineage metadata for loaded dataset."""
    fmt = next(iter(_data_loader_module.HASH_LOADER_REGISTRY))
    snapshot_dir = tmp_path / "snapshot-123"
    snapshot_dir.mkdir()

    dataset = _dataset_stub(
        ref=str(tmp_path),
        name="booking_context_features",
        version="v1",
        fmt=fmt,
        path_suffix="features.{format}",
        merge_key="row_id",
    )
    cfg = SimpleNamespace(data=[dataset])

    dataset_file = snapshot_dir / f"features.{fmt}"
    dataset_file.write_text("placeholder", encoding="utf-8")

    df_loaded = pd.DataFrame({"row_id": [1, 2], "feature_a": [10.0, 20.0]})

    monkeypatch.setattr(
        "ml.feature_freezing.utils.data_loader.get_latest_snapshot_path",
        lambda path: snapshot_dir,
    )
    monkeypatch.setattr(
        "ml.feature_freezing.utils.data_loader.read_data",
        lambda _fmt, _path: df_loaded,
    )
    monkeypatch.setattr(
        "ml.feature_freezing.utils.data_loader.merge_dataset_into_main",
        lambda **kwargs: (kwargs["df"], "data-hash-123"),
    )
    monkeypatch.setitem(_data_loader_module.HASH_LOADER_REGISTRY, fmt, lambda path: "loader-hash-abc")

    merged, lineage = load_data_with_lineage(_as_any_config(cfg), snapshot_binding_key=None)

    assert merged.equals(df_loaded)
    assert len(lineage) == 1
    entry = lineage[0]
    assert entry.name == "booking_context_features"
    assert entry.snapshot_id == "snapshot-123"
    assert entry.loader_validation_hash == "loader-hash-abc"
    assert entry.data_hash == "data-hash-123"
    assert entry.row_count == 2
    assert entry.column_count == 2
