"""Unit tests for tabular metadata pipeline step orchestration."""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture()
def metadata_module(monkeypatch: pytest.MonkeyPatch):
    """Import metadata step module with heavy dependencies stubbed out."""
    sys.modules.pop(
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.metadata",
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
    deps_module = cast(Any, types.ModuleType("ml.feature_freezing.persistence.get_deps"))
    hash_df_module = cast(Any, types.ModuleType("ml.features.hashing.hash_dataframe_content"))
    hash_schema_module = cast(Any, types.ModuleType("ml.features.hashing.hash_feature_schema"))
    git_module = cast(Any, types.ModuleType("ml.utils.git"))
    hash_file_module = cast(Any, types.ModuleType("ml.utils.hashing.service"))
    runtime_module = cast(Any, types.ModuleType("ml.utils.runtime.runtime_info"))

    persistence_module.create_metadata = lambda **kwargs: {"metadata": "ok"}
    context_module.FreezeContext = object
    deps_module.get_deps = lambda: {"numpy": "2.0"}
    hash_df_module.hash_dataframe_content = lambda df: "in-memory-hash"
    hash_schema_module.hash_feature_schema = lambda df: "schema-hash"
    git_module.get_git_commit = lambda path: "abc123"
    hash_file_module.hash_file = lambda path: "file-hash"
    runtime_module.get_runtime_info = lambda: {"os": "Linux"}

    monkeypatch.setitem(sys.modules, "ml.feature_freezing.freeze_strategies.tabular.persistence", persistence_module)
    monkeypatch.setitem(sys.modules, "ml.feature_freezing.freeze_strategies.tabular.pipeline.context", context_module)
    monkeypatch.setitem(sys.modules, "ml.feature_freezing.persistence.get_deps", deps_module)
    monkeypatch.setitem(sys.modules, "ml.features.hashing.hash_dataframe_content", hash_df_module)
    monkeypatch.setitem(sys.modules, "ml.features.hashing.hash_feature_schema", hash_schema_module)
    monkeypatch.setitem(sys.modules, "ml.utils.git", git_module)
    monkeypatch.setitem(sys.modules, "ml.utils.hashing.service", hash_file_module)
    monkeypatch.setitem(sys.modules, "ml.utils.runtime.runtime_info", runtime_module)

    return importlib.import_module(
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.metadata"
    )


class _Ctx:
    """Minimal context stub exposing attributes/properties consumed by MetadataStep."""

    def __init__(self, *, operators: Any | None) -> None:
        self.config = SimpleNamespace(operators=operators)
        self.start_time = 10.0
        self.timestamp = "2026-03-05T00:00:00"
        self.owner = "tests"

        self._features = pd.DataFrame({"row_id": [1], "x": [2.0]})
        self._data_path = Path("features.parquet")
        self._snapshot_path = Path("snapshot")
        self._schema_path = Path("schema")
        self._lineage = [SimpleNamespace(a=1), SimpleNamespace(b=2)]

        self.config_hash: str | None = None
        self.metadata: dict | None = None

    @property
    def require_features(self) -> pd.DataFrame:
        return self._features

    @property
    def require_data_path(self) -> Path:
        return self._data_path

    @property
    def require_snapshot_path(self) -> Path:
        return self._snapshot_path

    @property
    def require_schema_path(self) -> Path:
        return self._schema_path

    @property
    def require_data_lineage(self) -> list[SimpleNamespace]:
        return self._lineage

    @property
    def require_config_hash(self) -> str:
        assert self.config_hash is not None
        return self.config_hash


def test_metadata_step_sets_config_hash_and_metadata_payload(metadata_module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compute hashes/runtime payload and store final metadata on context."""
    captured: dict[str, Any] = {}

    def _create_metadata(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return {"saved": True}

    monkeypatch.setattr(metadata_module, "create_metadata", _create_metadata)
    monkeypatch.setattr(metadata_module.time, "perf_counter", lambda: 12.345)

    step = metadata_module.MetadataStep(hash_config=lambda cfg: "cfg-hash")
    ctx = _Ctx(operators=None)

    out = step.run(ctx)

    assert out is ctx
    assert ctx.config_hash == "cfg-hash"
    assert ctx.metadata == {"saved": True}
    assert captured["operator_hash"] == "none"
    assert captured["config_hash"] == "cfg-hash"
    assert captured["duration"] == 2.345
    assert captured["data_lineage"] == [{"a": 1}, {"b": 2}]


def test_metadata_step_uses_operator_hash_when_operators_present(metadata_module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use configured operator hash in metadata when operators block is present."""
    captured: dict[str, Any] = {}

    def _create_metadata(**kwargs: Any) -> dict:
        captured.update(kwargs)
        return {"saved": True}

    monkeypatch.setattr(metadata_module, "create_metadata", _create_metadata)
    monkeypatch.setattr(metadata_module.time, "perf_counter", lambda: 11.0)

    step = metadata_module.MetadataStep(hash_config=lambda cfg: "cfg-hash")
    ctx = _Ctx(operators=SimpleNamespace(hash="op-hash-123"))

    step.run(ctx)

    assert captured["operator_hash"] == "op-hash-123"
