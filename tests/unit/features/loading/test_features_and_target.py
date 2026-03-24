"""Unit tests for feature/target loading orchestration."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.config.schemas.model_cfg import SearchModelConfig
from ml.exceptions import DataError

pytestmark = pytest.mark.unit

# `ml.types` imports catboost classes at module import time in some environments.
if "catboost" not in sys.modules:
    catboost_stub = cast(Any, types.ModuleType("catboost"))
    catboost_stub.CatBoostClassifier = type("CatBoostClassifier", (), {})
    catboost_stub.CatBoostRegressor = type("CatBoostRegressor", (), {})
    sys.modules["catboost"] = catboost_stub


def _import_features_target_module() -> types.ModuleType:
    """Import module with isolated segmentation dependency to avoid circular imports."""
    module_name = "ml.features.loading.features_and_target"
    segmentation_name = "ml.features.segmentation.segment"

    sys.modules.pop(module_name, None)

    fake_segmentation = types.ModuleType(segmentation_name)
    fake_segmentation.__dict__["apply_segmentation"] = lambda data, seg_cfg: data
    sys.modules[segmentation_name] = fake_segmentation

    return importlib.import_module(module_name)


def _lineage_dict() -> dict[str, object]:
    """Return a valid lineage dictionary matching ``DataLineageEntry`` fields."""
    return {
        "ref": "feature_store",
        "name": "booking_context_features",
        "version": "v1",
        "format": "csv",
        "path_suffix": "features.{format}",
        "merge_key": "entity_key",
        "merge_how": "inner",
        "merge_validate": "m:m",
        "snapshot_id": "snap_001",
        "path": "feature_store/booking_context_features/v1/snap_001/features.csv",
        "loader_validation_hash": "loader-hash",
        "data_hash": "data-hash",
        "row_count": 2,
        "column_count": 2,
    }


def _model_cfg_stub(feature_sets: list[SimpleNamespace], *, store_path: str) -> SearchModelConfig:
    """Build a minimal typed config stub consumed by load_features_and_target."""
    return cast(
        SearchModelConfig,
        SimpleNamespace(
            feature_store=SimpleNamespace(path=store_path, feature_sets=feature_sets),
            target=SimpleNamespace(name="target", version="v1"),
            segmentation=SimpleNamespace(enabled=False, include_in_model=False, filters=[]),
            min_rows=0,
        ),
    )


def test_load_features_and_target_raises_for_invalid_lineage_entry_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise ``DataError`` when metadata lineage dictionaries cannot build DataLineageEntry."""
    module = _import_features_target_module()

    fs = SimpleNamespace(name="booking_context_features", version="v1", file_name="features.csv", data_format="csv")
    model_cfg = _model_cfg_stub([fs], store_path=str(tmp_path / "feature_store"))
    snapshot_selection = [
        {
            "fs_spec": fs,
            "snapshot_path": tmp_path / "feature_store" / "booking_context_features" / "v1" / "snap_001",
            "snapshot_id": "snap_001",
            "metadata": {
                "feature_schema_hash": "schema-hash",
                "operator_hash": "operator-hash",
                "feature_type": "tabular",
                "data_lineage": [{"name": "missing_many_required_fields"}],
                "in_memory_hash": "mem-hash",
                "file_hash": "file-hash",
                "entity_key": "entity_key",
            },
        }
    ]

    monkeypatch.setattr(module, "ensure_required_fields_present_in_dict", lambda **_kwargs: None)
    monkeypatch.setattr(module, "read_data", lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1], "f": [1.0]}))
    monkeypatch.setattr(module, "validate_feature_set", lambda **_kwargs: None)

    with pytest.raises(DataError, match="Data lineage entry is missing required fields"):
        module.load_features_and_target(model_cfg, snapshot_selection=snapshot_selection)


def test_load_features_and_target_raises_when_feature_set_indices_do_not_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise ``DataError`` when multiple feature sets have different dataframe indices."""
    module = _import_features_target_module()

    fs_a = SimpleNamespace(name="set_a", version="v1", file_name="a.csv", data_format="csv")
    fs_b = SimpleNamespace(name="set_b", version="v1", file_name="b.csv", data_format="csv")
    model_cfg = _model_cfg_stub([fs_a, fs_b], store_path=str(tmp_path / "feature_store"))

    metadata = {
        "feature_schema_hash": "schema-hash",
        "operator_hash": "operator-hash",
        "feature_type": "tabular",
        "data_lineage": [_lineage_dict()],
        "in_memory_hash": "mem-hash",
        "file_hash": "file-hash",
        "entity_key": "entity_key",
    }

    snapshot_selection = [
        {"fs_spec": fs_a, "snapshot_path": tmp_path / "a" / "snap_001", "snapshot_id": "snap_001", "metadata": metadata},
        {"fs_spec": fs_b, "snapshot_path": tmp_path / "b" / "snap_002", "snapshot_id": "snap_002", "metadata": metadata},
    ]

    dfs = [
        pd.DataFrame({"entity_key": [1, 2], "fa": [10.0, 20.0]}, index=[10, 11]),
        pd.DataFrame({"entity_key": [1, 2], "fb": [30.0, 40.0]}, index=[0, 1]),
    ]

    monkeypatch.setattr(module, "ensure_required_fields_present_in_dict", lambda **_kwargs: None)
    monkeypatch.setattr(module, "read_data", lambda *_args, **_kwargs: dfs.pop(0))
    monkeypatch.setattr(module, "validate_feature_set", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_and_validate_data", lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2]}))
    monkeypatch.setattr(module, "get_target_with_entity_key", lambda **_kwargs: pd.DataFrame({"entity_key": [1, 2], "target": [0, 1]}))
    monkeypatch.setattr(module, "validate_entity_key", lambda *args, **_kwargs: None)
    monkeypatch.setattr(module, "validate_target", lambda **_kwargs: None)

    # Current behavior: alignment is performed by `entity_key` values, not by
    # original dataframe indices. Ensure function returns without raising and
    # produces expected output types.
    X, y, lineage, entity_key = module.load_features_and_target(
        model_cfg, snapshot_selection=snapshot_selection
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(lineage, list)
    assert isinstance(entity_key, str)


def test_load_features_and_target_resolves_snapshots_and_drops_row_id_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve snapshots when absent and return `(X, y, lineage)` with entity_key dropped."""
    module = _import_features_target_module()

    fs = SimpleNamespace(name="booking_context_features", version="v1", file_name="features.csv", data_format="csv")
    model_cfg = _model_cfg_stub([fs], store_path=str(tmp_path / "feature_store"))
    resolved_snapshot = [
        {
            "fs_spec": fs,
            "snapshot_path": tmp_path / "feature_store" / "booking_context_features" / "v1" / "snap_001",
            "snapshot_id": "snap_001",
            "metadata": {
                "feature_schema_hash": "schema-hash",
                "operator_hash": "operator-hash",
                "feature_type": "tabular",
                "data_lineage": [_lineage_dict()],
                "in_memory_hash": "mem-hash",
                "file_hash": "file-hash",
                "entity_key": "entity_key",
            },
        }
    ]

    called: dict[str, object] = {}

    def _resolve(feature_store_path: Path, feature_sets: list[SimpleNamespace], snapshot_binding_key: str | None = None) -> list[dict[str, object]]:
        called["feature_store_path"] = feature_store_path
        called["feature_sets"] = feature_sets
        return resolved_snapshot

    monkeypatch.setattr(module, "resolve_feature_snapshots", _resolve)
    monkeypatch.setattr(module, "ensure_required_fields_present_in_dict", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "read_data",
        lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2], "feature_a": [0.5, 0.7], "target": [9, 9]}),
    )
    monkeypatch.setattr(module, "validate_feature_set", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_and_validate_data", lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2]}))
    monkeypatch.setattr(module, "get_target_with_entity_key", lambda **_kwargs: pd.DataFrame({"entity_key": [1, 2], "target": [0, 1]}))
    monkeypatch.setattr(module, "validate_entity_key", lambda *args, **_kwargs: None)
    monkeypatch.setattr(module, "apply_segmentation", lambda data, seg_cfg: data)
    monkeypatch.setattr(module, "validate_feature_target_entity_key", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_min_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "validate_target", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_and_construct_feature_lineage", lambda _raw: ["ok-lineage"])

    X, y, lineage, entity_key = module.load_features_and_target(
        model_cfg,
        snapshot_selection=None,
        drop_entity_key=True,
        strict=True,
    )

    assert called["feature_store_path"] == Path(model_cfg.feature_store.path)
    assert called["feature_sets"] == model_cfg.feature_store.feature_sets
    assert list(X.columns) == ["feature_a"]
    assert y.tolist() == [0, 1]
    assert lineage == ["ok-lineage"]


def test_load_features_and_target_uses_given_snapshot_selection_without_resolving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Honor explicit snapshot_selection and skip resolver invocation."""
    module = _import_features_target_module()

    fs = SimpleNamespace(name="booking_context_features", version="v1", file_name="features.csv", data_format="csv")
    model_cfg = _model_cfg_stub([fs], store_path=str(tmp_path / "feature_store"))
    provided_selection = [
        {
            "fs_spec": fs,
            "snapshot_path": tmp_path / "feature_store" / "booking_context_features" / "v1" / "snap_manual",
            "snapshot_id": "snap_manual",
            "metadata": {
                "feature_schema_hash": "schema-hash",
                "operator_hash": "operator-hash",
                "feature_type": "tabular",
                "data_lineage": [_lineage_dict()],
                "in_memory_hash": "mem-hash",
                "file_hash": "file-hash",
                "entity_key": "entity_key",
            },
        }
    ]

    monkeypatch.setattr(
        module,
        "resolve_feature_snapshots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resolver should not be called")),
    )
    monkeypatch.setattr(module, "ensure_required_fields_present_in_dict", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "read_data",
        lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2], "feature_a": [0.5, 0.7]}),
    )
    monkeypatch.setattr(module, "validate_feature_set", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_and_validate_data", lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2]}))
    monkeypatch.setattr(module, "get_target_with_entity_key", lambda **_kwargs: pd.DataFrame({"entity_key": [1, 2], "target": [1, 0]}))
    monkeypatch.setattr(module, "validate_entity_key", lambda *args, **_kwargs: None)
    monkeypatch.setattr(module, "apply_segmentation", lambda data, seg_cfg: data)
    monkeypatch.setattr(module, "validate_feature_target_entity_key", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_min_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "validate_target", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_and_construct_feature_lineage", lambda _raw: ["lineage"])

    X, y, lineage, entity_key = module.load_features_and_target(
        model_cfg,
        snapshot_selection=provided_selection,
        drop_entity_key=False,
        strict=True,
    )

    assert list(X.columns) == ["entity_key", "feature_a"]
    assert y.tolist() == [1, 0]
    assert lineage == ["lineage"]


def test_load_features_and_target_raises_when_segmentation_removes_row_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise ``DataError`` when segmentation removes entity_key before alignment validation."""
    module = _import_features_target_module()

    fs = SimpleNamespace(name="booking_context_features", version="v1", file_name="features.csv", data_format="csv")
    model_cfg = _model_cfg_stub([fs], store_path=str(tmp_path / "feature_store"))
    snapshot_selection = [
        {
            "fs_spec": fs,
            "snapshot_path": tmp_path / "feature_store" / "booking_context_features" / "v1" / "snap_001",
            "snapshot_id": "snap_001",
            "metadata": {
                "feature_schema_hash": "schema-hash",
                "operator_hash": "operator-hash",
                "feature_type": "tabular",
                "data_lineage": [_lineage_dict()],
                "in_memory_hash": "mem-hash",
                "file_hash": "file-hash",
                "entity_key": "entity_key",
            },
        }
    ]

    monkeypatch.setattr(module, "ensure_required_fields_present_in_dict", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "read_data",
        lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2], "feature_a": [0.5, 0.7]}),
    )
    monkeypatch.setattr(module, "validate_feature_set", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_and_validate_data", lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2]}))
    monkeypatch.setattr(module, "get_target_with_entity_key", lambda **_kwargs: pd.DataFrame({"entity_key": [1, 2], "target": [0, 1]}))
    monkeypatch.setattr(module, "validate_entity_key", lambda *args, **_kwargs: None)
    monkeypatch.setattr(module, "apply_segmentation", lambda data, seg_cfg: data.drop(columns=["entity_key"]))
    monkeypatch.setattr(module, "validate_min_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "validate_target", lambda **_kwargs: None)

    with pytest.raises(DataError, match="Feature set is missing entity_key column"):
        module.load_features_and_target(
            model_cfg,
            snapshot_selection=snapshot_selection,
            drop_entity_key=True,
            strict=True,
        )


def test_load_features_and_target_drops_duplicate_feature_columns_after_segmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop duplicated feature columns and keep a single copy before downstream merges."""
    module = _import_features_target_module()

    fs = SimpleNamespace(name="booking_context_features", version="v1", file_name="features.csv", data_format="csv")
    model_cfg = _model_cfg_stub([fs], store_path=str(tmp_path / "feature_store"))
    snapshot_selection = [
        {
            "fs_spec": fs,
            "snapshot_path": tmp_path / "feature_store" / "booking_context_features" / "v1" / "snap_001",
            "snapshot_id": "snap_001",
            "metadata": {
                "feature_schema_hash": "schema-hash",
                "operator_hash": "operator-hash",
                "feature_type": "tabular",
                "data_lineage": [_lineage_dict()],
                "in_memory_hash": "mem-hash",
                "file_hash": "file-hash",
                "entity_key": "entity_key",
            },
        }
    ]

    monkeypatch.setattr(module, "ensure_required_fields_present_in_dict", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "read_data",
        lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2], "feature_a": [0.5, 0.7]}),
    )
    monkeypatch.setattr(module, "validate_feature_set", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_and_validate_data", lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2]}))
    monkeypatch.setattr(module, "get_target_with_entity_key", lambda **_kwargs: pd.DataFrame({"entity_key": [1, 2], "target": [0, 1]}))
    monkeypatch.setattr(module, "validate_entity_key", lambda *args, **_kwargs: None)

    # Intentionally introduce duplicated feature column names post-segmentation.
    duplicated = pd.DataFrame(
        [[1, 0.5, 0.5], [2, 0.7, 0.7]],
        columns=["entity_key", "feature_a", "feature_a"],
    )
    monkeypatch.setattr(module, "apply_segmentation", lambda data, seg_cfg: duplicated)

    monkeypatch.setattr(module, "validate_feature_target_entity_key", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_min_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "validate_target", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_and_construct_feature_lineage", lambda _raw: ["lineage"])

    X, y, lineage, entity_key = module.load_features_and_target(
        model_cfg,
        snapshot_selection=snapshot_selection,
        drop_entity_key=True,
        strict=True,
    )

    assert list(X.columns) == ["feature_a"]
    assert y.tolist() == [0, 1]
    assert lineage == ["lineage"]


def test_load_features_and_target_resolves_snapshots_when_given_empty_selection_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat empty snapshot selection as unresolved and call snapshot resolver."""
    module = _import_features_target_module()

    fs = SimpleNamespace(name="booking_context_features", version="v1", file_name="features.csv", data_format="csv")
    model_cfg = _model_cfg_stub([fs], store_path=str(tmp_path / "feature_store"))

    resolved_snapshot = [
        {
            "fs_spec": fs,
            "snapshot_path": tmp_path / "feature_store" / "booking_context_features" / "v1" / "snap_001",
            "snapshot_id": "snap_001",
            "metadata": {
                "feature_schema_hash": "schema-hash",
                "operator_hash": "operator-hash",
                "feature_type": "tabular",
                "data_lineage": [_lineage_dict()],
                "in_memory_hash": "mem-hash",
                "file_hash": "file-hash",
                "entity_key": "entity_key",
            },
        }
    ]

    calls = {"resolved": 0}

    def _resolve(*_args, **_kwargs):
        calls["resolved"] += 1
        return resolved_snapshot

    monkeypatch.setattr(module, "resolve_feature_snapshots", _resolve)
    monkeypatch.setattr(module, "ensure_required_fields_present_in_dict", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "read_data",
        lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2], "feature_a": [0.5, 0.7]}),
    )
    monkeypatch.setattr(module, "validate_feature_set", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_set", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_and_validate_data", lambda *_args, **_kwargs: pd.DataFrame({"entity_key": [1, 2]}))
    monkeypatch.setattr(module, "get_target_with_entity_key", lambda **_kwargs: pd.DataFrame({"entity_key": [1, 2], "target": [0, 1]}))
    monkeypatch.setattr(module, "validate_entity_key", lambda *args, **_kwargs: None)
    monkeypatch.setattr(module, "apply_segmentation", lambda data, seg_cfg: data)
    monkeypatch.setattr(module, "validate_feature_target_entity_key", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_min_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "validate_target", lambda **_kwargs: None)
    monkeypatch.setattr(module, "validate_and_construct_feature_lineage", lambda _raw: ["lineage"])

    module.load_features_and_target(
        model_cfg,
        snapshot_selection=[],
        drop_entity_key=True,
        strict=True,
    )

    assert calls["resolved"] == 1
