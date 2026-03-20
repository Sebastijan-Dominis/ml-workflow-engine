"""Unit tests for tabular ingestion pipeline step orchestration."""

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture()
def ingestion_module(monkeypatch: pytest.MonkeyPatch):
    """Import ingestion step module with heavy dependencies stubbed out."""
    sys.modules.pop(
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.ingestion",
        None,
    )

    min_rows_module = cast(Any, types.ModuleType("ml.data.validation.validate_min_rows"))
    row_id_module = cast(Any, types.ModuleType("ml.data.validation.validate_row_id"))
    data_loader_module = cast(Any, types.ModuleType("ml.feature_freezing.utils.data_loader"))
    operators_module = cast(Any, types.ModuleType("ml.feature_freezing.utils.operators"))
    context_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.pipeline.context"),
    )

    min_rows_module.validate_min_rows = lambda data, min_rows: None
    row_id_module.validate_row_id = lambda data: None
    # accept optional snapshot_binding_key for compatibility with updated signature
    data_loader_module.load_data_with_lineage = lambda *args, **kwargs: (pd.DataFrame(), [])
    operators_module.validate_operators = lambda names, op_hash: None
    context_module.FreezeContext = object

    monkeypatch.setitem(sys.modules, "ml.data.validation.validate_min_rows", min_rows_module)
    monkeypatch.setitem(sys.modules, "ml.data.validation.validate_row_id", row_id_module)
    monkeypatch.setitem(sys.modules, "ml.feature_freezing.utils.data_loader", data_loader_module)
    monkeypatch.setitem(sys.modules, "ml.feature_freezing.utils.operators", operators_module)
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.context",
        context_module,
    )

    return importlib.import_module(
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.ingestion"
    )


def test_ingestion_step_loads_data_validates_and_sets_context(ingestion_module) -> None:
    """Load data, run baseline validators, and attach data/lineage to context."""
    step = ingestion_module.IngestionStep()
    loaded_df = pd.DataFrame({"row_id": [1, 2], "x": [10, 20]})
    loaded_lineage = [cast(Any, object())]

    called = {"row_id": False, "min_rows": False, "operators": False}

    ingestion_module.load_data_with_lineage = lambda *args, **kwargs: (loaded_df, loaded_lineage)

    def _validate_row_id(data: pd.DataFrame) -> None:
        called["row_id"] = True
        assert data is loaded_df

    def _validate_min_rows(data: pd.DataFrame, min_rows: int) -> None:
        called["min_rows"] = True
        assert data is loaded_df
        assert min_rows == 100

    def _validate_operators(names: list[str], op_hash: str) -> None:
        called["operators"] = True
        assert names == ["op1"]
        assert op_hash == "hash-1"

    ingestion_module.validate_row_id = _validate_row_id
    ingestion_module.validate_min_rows = _validate_min_rows
    ingestion_module.validate_operators = _validate_operators

    ctx = SimpleNamespace(
        config=SimpleNamespace(min_rows=100, operators=SimpleNamespace(names=["op1"], hash="hash-1")),
        data=None,
        data_lineage=None,
        snapshot_binding_key="snapshot_key_123",
    )

    out = step.run(ctx)

    assert out is ctx
    assert called == {"row_id": True, "min_rows": True, "operators": True}
    assert ctx.data is loaded_df
    assert ctx.data_lineage is loaded_lineage


def test_ingestion_step_skips_operator_validation_when_no_operators(ingestion_module) -> None:
    """Skip operator hash validation when config has no operator block."""
    step = ingestion_module.IngestionStep()
    loaded_df = pd.DataFrame({"row_id": [1], "x": [10]})

    called = {"operators": False}

    ingestion_module.load_data_with_lineage = lambda *args, **kwargs: (loaded_df, [])
    ingestion_module.validate_row_id = lambda data: None
    ingestion_module.validate_min_rows = lambda data, min_rows: None

    def _validate_operators(names: list[str], op_hash: str) -> None:
        called["operators"] = True

    ingestion_module.validate_operators = _validate_operators

    ctx = SimpleNamespace(
        config=SimpleNamespace(min_rows=1, operators=None),
        data=None,
        data_lineage=None,
        snapshot_binding_key="snapshot_key_123",
    )

    step.run(ctx)

    assert called["operators"] is False
