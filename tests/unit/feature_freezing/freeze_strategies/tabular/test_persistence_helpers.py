"""Unit tests for tabular freeze persistence and metadata helper functions."""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


class _DerivedOperator:
    """Test operator that emits a deterministic derived feature."""

    output_features = ["derived_a"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["derived_a"] = X["x"] + 10
        return X


class _ParquetRecorder:
    """Capture to_parquet invocations for assertion without disk IO dependencies."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, bool, str | None]] = []

    def __call__(self, path: Path, *, index: bool, compression: str | None = None) -> None:
        self.calls.append((path, index, compression))


@pytest.fixture()
def persistence_module(monkeypatch: pytest.MonkeyPatch):
    """Import persistence helper module with registry dependencies stubbed."""
    sys.modules.pop("ml.feature_freezing.freeze_strategies.tabular.persistence", None)

    registries_pkg = cast(Any, types.ModuleType("ml.registries"))
    catalogs_stub = cast(Any, types.ModuleType("ml.registries.catalogs"))
    catalogs_stub.FEATURE_OPERATORS = {"derived_op": _DerivedOperator}
    registries_pkg.catalogs = catalogs_stub

    monkeypatch.setitem(sys.modules, "ml.registries", registries_pkg)
    monkeypatch.setitem(sys.modules, "ml.registries.catalogs", catalogs_stub)

    return importlib.import_module("ml.feature_freezing.freeze_strategies.tabular.persistence")


def test_persist_feature_snapshot_creates_snapshot_dir_and_returns_paths(
    persistence_module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create snapshot directory and return both snapshot and data artifact paths."""
    cfg = cast(
        Any,
        SimpleNamespace(
            feature_store_path=str(tmp_path / "feature_store"),
            storage=SimpleNamespace(format="parquet", compression="snappy"),
        ),
    )
    features = pd.DataFrame({"row_id": [1], "x": [2]})

    def _fake_freeze_parquet(path: Path, *, features: pd.DataFrame, compression: str | None = None) -> Path:
        return path / "features.parquet"

    monkeypatch.setattr(persistence_module, "freeze_parquet", _fake_freeze_parquet)

    snapshot_path, data_path = persistence_module.persist_feature_snapshot(
        cfg,
        features=features,
        snapshot_id="snapshot-1",
    )

    assert snapshot_path.exists()
    assert data_path == snapshot_path / "features.parquet"


def test_freeze_parquet_writes_expected_output_path(
    persistence_module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write parquet to the canonical snapshot filename and return that path."""
    recorder = _ParquetRecorder()
    monkeypatch.setattr(pd.DataFrame, "to_parquet", recorder)

    features = pd.DataFrame({"x": [1, 2]})
    out_path = persistence_module.freeze_parquet(
        tmp_path,
        features=features,
        compression="snappy",
    )

    assert out_path == tmp_path / "features.parquet"
    assert recorder.calls == [(tmp_path / "features.parquet", False, "snappy")]


def test_save_input_schema_writes_csv_with_expected_columns(
    persistence_module,
    tmp_path: Path,
) -> None:
    """Persist input schema with feature, dtype, and role columns."""
    features = pd.DataFrame({"x": pd.Series([1], dtype="int64")})

    persistence_module.save_input_schema(tmp_path, features)

    schema = pd.read_csv(tmp_path / "input_schema.csv")
    assert schema.columns.tolist() == ["feature", "dtype", "role"]
    assert schema.iloc[0]["feature"] == "x"
    assert schema.iloc[0]["role"] == "input"


def test_save_input_schema_skips_when_schema_already_exists(
    persistence_module,
    tmp_path: Path,
) -> None:
    """Leave existing input schema untouched to keep writes idempotent."""
    schema_path = tmp_path / "input_schema.csv"
    schema_path.write_text("feature,dtype,role\npreexisting,int64,input\n", encoding="utf-8")

    persistence_module.save_input_schema(tmp_path, pd.DataFrame({"x": [1]}))

    persisted = pd.read_csv(schema_path)
    assert persisted.iloc[0]["feature"] == "preexisting"


def test_save_derived_schema_writes_operator_outputs(
    persistence_module,
    tmp_path: Path,
) -> None:
    """Persist derived schema rows including source operator and materialization flag."""
    features = pd.DataFrame({"x": [1, 2]})

    persistence_module.save_derived_schema(
        tmp_path,
        features=features,
        operator_names=["derived_op"],
        mode="materialized",
    )

    schema = pd.read_csv(tmp_path / "derived_schema.csv")
    assert schema.iloc[0]["feature"] == "derived_a"
    assert schema.iloc[0]["source_operator"] == "_DerivedOperator"
    assert bool(schema.iloc[0]["materialized"]) is True


def test_save_derived_schema_skips_when_schema_already_exists(
    persistence_module,
    tmp_path: Path,
) -> None:
    """Preserve existing derived schema file when rerunning persistence."""
    schema_path = tmp_path / "derived_schema.csv"
    schema_path.write_text(
        "feature,dtype,role,source_operator,materialized\nold,float64,derived,OldOp,False\n",
        encoding="utf-8",
    )

    persistence_module.save_derived_schema(
        tmp_path,
        features=pd.DataFrame({"x": [1, 2]}),
        operator_names=["derived_op"],
        mode="materialized",
    )

    persisted = pd.read_csv(schema_path)
    assert persisted.iloc[0]["feature"] == "old"


def test_persist_feature_snapshot_raises_for_unknown_storage_format(
    persistence_module,
    tmp_path: Path,
) -> None:
    """Fail clearly when configured storage format is not in the registry."""
    cfg = cast(
        Any,
        SimpleNamespace(
            feature_store_path=str(tmp_path / "feature_store"),
            storage=SimpleNamespace(format="csv", compression=None),
        ),
    )

    with pytest.raises(KeyError, match="csv"):
        persistence_module.persist_feature_snapshot(
            cfg,
            features=pd.DataFrame({"row_id": [1], "x": [2]}),
            snapshot_id="snapshot-unknown",
        )


def test_create_metadata_returns_validated_model_dump(
    persistence_module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return metadata via validate_freeze_metadata.model_dump passthrough."""

    class _Validated:
        def model_dump(self, exclude_none: bool = True) -> dict:
            return {"validated": True, "exclude_none": exclude_none}

    monkeypatch.setattr(
        persistence_module,
        "validate_freeze_metadata",
        lambda payload: _Validated(),
    )

    metadata = persistence_module.create_metadata(
        timestamp="2026-03-05T00:00:00",
        snapshot_path=tmp_path / "snapshot-1",
        schema_path=tmp_path / "input_schema.csv",
        data_lineage=[{"name": "ds"}],
        in_memory_hash="mem-h",
        file_hash="file-h",
        operator_hash="op-h",
        config_hash="cfg-h",
        feature_schema_hash="schema-h",
        runtime={"py": "3.11"},
        features=pd.DataFrame({"x": [1, 2]}),
        duration=1.234,
        owner="tests",
    )

    assert metadata == {"validated": True, "exclude_none": True}
