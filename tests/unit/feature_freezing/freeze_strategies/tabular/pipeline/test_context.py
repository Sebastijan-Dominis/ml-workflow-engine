"""Unit tests for tabular freeze pipeline context required-value accessors."""

import importlib
import sys
import types
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from ml.exceptions import RuntimeMLError

pytestmark = pytest.mark.unit

# FreezeContext imports ml.types, which imports catboost. Stub to keep tests isolated.
if "catboost" not in sys.modules:
    catboost_stub = cast(Any, types.ModuleType("catboost"))
    catboost_stub.CatBoostClassifier = type("CatBoostClassifier", (), {})
    catboost_stub.CatBoostRegressor = type("CatBoostRegressor", (), {})
    sys.modules["catboost"] = catboost_stub

FreezeContext = importlib.import_module(
    "ml.feature_freezing.freeze_strategies.tabular.pipeline.context"
).FreezeContext


def _ctx_stub() -> Any:
    """Create minimal FreezeContext instance with required constructor fields."""
    return FreezeContext(
        config=cast(Any, object()),
        timestamp="2026-03-05T00:00:00",
        snapshot_id="snapshot-1",
        start_time=0.0,
        owner="tests",
    )


@pytest.mark.parametrize(
    ("field_name", "property_name", "error_fragment"),
    [
        ("data", "require_data", "Data not loaded yet"),
        ("data_lineage", "require_data_lineage", "Data lineage not computed yet"),
        ("features", "require_features", "Features not prepared yet"),
        ("snapshot_path", "require_snapshot_path", "Snapshot not persisted yet"),
        ("schema_path", "require_schema_path", "Schema not persisted yet"),
        ("data_path", "require_data_path", "Data path not set"),
        ("config_hash", "require_config_hash", "Config hash not computed yet"),
        ("metadata", "require_metadata", "Metadata not created yet"),
    ],
)
def test_context_require_properties_raise_when_unset(
    field_name: str,
    property_name: str,
    error_fragment: str,
) -> None:
    """Raise RuntimeMLError when required context value has not been populated."""
    ctx = _ctx_stub()
    setattr(ctx, field_name, None)

    with pytest.raises(RuntimeMLError, match=error_fragment):
        getattr(ctx, property_name)


def test_context_require_properties_return_values_when_set() -> None:
    """Return previously-populated values for all required context properties."""
    ctx = _ctx_stub()

    data = pd.DataFrame({"row_id": [1]})
    lineage = [cast(Any, object())]
    features = pd.DataFrame({"f": [1.0]})
    snapshot_path = Path("snapshot")
    schema_path = Path("schema.yaml")
    data_path = Path("data.parquet")
    config_hash = "abc123"
    metadata = {"ok": True}

    ctx.data = data
    ctx.data_lineage = lineage
    ctx.features = features
    ctx.snapshot_path = snapshot_path
    ctx.schema_path = schema_path
    ctx.data_path = data_path
    ctx.config_hash = config_hash
    ctx.metadata = metadata

    assert ctx.require_data is data
    assert ctx.require_data_lineage is lineage
    assert ctx.require_features is features
    assert ctx.require_snapshot_path is snapshot_path
    assert ctx.require_schema_path is schema_path
    assert ctx.require_data_path is data_path
    assert ctx.require_config_hash == config_hash
    assert ctx.require_metadata is metadata
