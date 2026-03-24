"""Unit tests for tabular persistence pipeline step orchestration."""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.exceptions import PersistenceError

pytestmark = pytest.mark.unit


@pytest.fixture()
def persistence_step_module(monkeypatch: pytest.MonkeyPatch):
    """Import persistence step module with persistence/context dependencies stubbed."""
    sys.modules.pop(
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.persistence",
        None,
    )

    persistence_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.persistence"),
    )
    context_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.pipeline.context"),
    )

    persistence_module.persist_feature_snapshot = lambda config, *, features, snapshot_id: (Path("snapshot"), Path("snapshot/features.parquet"))
    persistence_module.save_input_schema = lambda path, features: None
    persistence_module.save_derived_schema = lambda path, *, features, operator_names, mode: None
    context_module.FreezeContext = object

    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.persistence",
        persistence_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.context",
        context_module,
    )

    return importlib.import_module(
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.persistence"
    )


def test_persistence_step_persists_artifacts_and_updates_context(persistence_step_module) -> None:
    """Persist snapshot/schemas and store resulting artifact paths on context."""
    step = persistence_step_module.PersistenceStep()

    features = pd.DataFrame({"entity_key": [1], "x": [2.0], "y": [3.0]})
    schema_save: dict[str, list[str] | None] = {"columns": None}

    persistence_step_module.persist_feature_snapshot = (
        lambda config, *, features, snapshot_id: (Path("snapshots/s1"), Path("snapshots/s1/features.parquet"))
    )

    def _save_input_schema(path: Path, frame: pd.DataFrame) -> None:
        schema_save["columns"] = frame.columns.tolist()

    persistence_step_module.save_input_schema = _save_input_schema
    persistence_step_module.save_derived_schema = lambda path, *, features, operator_names, mode: None

    ctx = SimpleNamespace(
        config=SimpleNamespace(feature_store_path=Path("feature_store"), operators=None, entity_key="entity_key"),
        snapshot_id="s1",
        require_features=features,
        snapshot_path=None,
        schema_path=None,
        data_path=None,
    )

    out = step.run(ctx)

    assert out is ctx
    assert schema_save["columns"] == ["x", "y"]
    assert ctx.snapshot_path == Path("snapshots/s1")
    assert ctx.schema_path == Path("feature_store")
    assert ctx.data_path == Path("snapshots/s1/features.parquet")


def test_persistence_step_raises_when_row_id_missing_from_features(persistence_step_module) -> None:
    """Reject persistence when required row_id column is absent from features."""
    step = persistence_step_module.PersistenceStep()
    ctx = SimpleNamespace(
        config=SimpleNamespace(feature_store_path=Path("feature_store"), operators=None, entity_key="entity_key"),
        snapshot_id="s1",
        require_features=pd.DataFrame({"x": [1]}),
    )

    with pytest.raises(PersistenceError, match="Expected 'entity_key' column"):
        step.run(ctx)


def test_persistence_step_wraps_input_schema_failures(persistence_step_module) -> None:
    """Wrap save_input_schema exceptions as PersistenceError with clear context."""
    step = persistence_step_module.PersistenceStep()

    persistence_step_module.persist_feature_snapshot = (
        lambda config, *, features, snapshot_id: (Path("snapshots/s1"), Path("snapshots/s1/features.parquet"))
    )

    def _raise_input_schema(path: Path, frame: pd.DataFrame) -> None:
        raise OSError("disk error")

    persistence_step_module.save_input_schema = _raise_input_schema

    ctx = SimpleNamespace(
        config=SimpleNamespace(feature_store_path=Path("feature_store"), operators=None, entity_key="entity_key"),
        snapshot_id="s1",
        require_features=pd.DataFrame({"entity_key": [1], "x": [2]}),
    )

    with pytest.raises(PersistenceError, match="Could not save input schema"):
        step.run(ctx)


def test_persistence_step_wraps_derived_schema_failures_when_operators_enabled(
    persistence_step_module,
) -> None:
    """Wrap save_derived_schema exceptions as PersistenceError when operators are configured."""
    step = persistence_step_module.PersistenceStep()

    persistence_step_module.persist_feature_snapshot = (
        lambda config, *, features, snapshot_id: (Path("snapshots/s1"), Path("snapshots/s1/features.parquet"))
    )
    persistence_step_module.save_input_schema = lambda path, frame: None

    def _raise_derived_schema(path: Path, *, features: pd.DataFrame, operator_names: list[str], mode: str) -> None:
        raise OSError("schema write failed")

    persistence_step_module.save_derived_schema = _raise_derived_schema

    ctx = SimpleNamespace(
        config=SimpleNamespace(
            feature_store_path=Path("feature_store"),
            operators=SimpleNamespace(names=["op"], mode="materialized"),
            entity_key="entity_key",
        ),
        snapshot_id="s1",
        require_features=pd.DataFrame({"entity_key": [1], "x": [2]}),
    )

    with pytest.raises(PersistenceError, match="Could not save derived schema"):
        step.run(ctx)
