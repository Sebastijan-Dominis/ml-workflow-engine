"""Unit tests for tabular freeze strategy orchestration."""

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.unit


@dataclass
class _FakeTabularConfig:
    """Minimal stand-in for TabularFeaturesConfig in strategy tests."""

    marker: str = "cfg"


class _FakeContext:
    """Context stub with required output accessors expected by strategy."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.snapshot_path = Path("snapshot/path")
        self.metadata = {"ok": True}

    @property
    def require_snapshot_path(self) -> Path:
        return self.snapshot_path

    @property
    def require_metadata(self) -> dict:
        return self.metadata


class _StepBase:
    """Base marker class for fake pipeline steps."""


class _FakeIngestion(_StepBase):
    pass


class _FakePreprocessing(_StepBase):
    pass


class _FakePersistence(_StepBase):
    pass


class _FakeMetadata(_StepBase):
    def __init__(self, hash_config: Any) -> None:
        self.hash_config = hash_config


@pytest.fixture()
def strategy_module(monkeypatch: pytest.MonkeyPatch):
    """Import FreezeTabular module with strategy dependencies stubbed."""
    sys.modules.pop("ml.feature_freezing.freeze_strategies.tabular.strategy", None)

    output_module = cast(Any, types.ModuleType("ml.feature_freezing.constants.output"))
    base_module = cast(Any, types.ModuleType("ml.feature_freezing.freeze_strategies.base"))
    validate_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.config.validate_feature_registry"),
    )
    config_models_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.config.models"),
    )
    context_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.pipeline.context"),
    )
    ingestion_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.ingestion"),
    )
    preprocessing_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.preprocessing"),
    )
    persistence_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.persistence"),
    )
    metadata_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.metadata"),
    )
    runner_module = cast(Any, types.ModuleType("ml.utils.pipeline_core.runner"))

    @dataclass
    class _FreezeOutput:
        snapshot_path: Path
        metadata: dict

    class _FreezeStrategy:
        @staticmethod
        def hash_config(config: Any) -> str:
            return "hash-from-base"

    class _Runner:
        @classmethod
        def __class_getitem__(cls, _item: Any):
            return cls

        def __init__(self, *, steps: list[Any]) -> None:
            self.steps = steps

        def run(self, ctx: Any) -> Any:
            return ctx

    output_module.FreezeOutput = _FreezeOutput
    base_module.FreezeStrategy = _FreezeStrategy
    validate_module.validate_feature_registry = lambda raw, strategy_type: _FakeTabularConfig(marker="validated")
    config_models_module.TabularFeaturesConfig = _FakeTabularConfig
    context_module.FreezeContext = _FakeContext
    ingestion_module.IngestionStep = _FakeIngestion
    preprocessing_module.PreprocessingStep = _FakePreprocessing
    persistence_module.PersistenceStep = _FakePersistence
    metadata_module.MetadataStep = _FakeMetadata
    runner_module.PipelineRunner = _Runner

    monkeypatch.setitem(sys.modules, "ml.feature_freezing.constants.output", output_module)
    monkeypatch.setitem(sys.modules, "ml.feature_freezing.freeze_strategies.base", base_module)
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.config.validate_feature_registry",
        validate_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.config.models",
        config_models_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.context",
        context_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.ingestion",
        ingestion_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.preprocessing",
        preprocessing_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.persistence",
        persistence_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.metadata",
        metadata_module,
    )
    monkeypatch.setitem(sys.modules, "ml.utils.pipeline_core.runner", runner_module)

    return importlib.import_module("ml.feature_freezing.freeze_strategies.tabular.strategy")


def test_freeze_tabular_uses_config_directly_when_already_validated(strategy_module) -> None:
    """Skip registry validation when config is already TabularFeaturesConfig."""
    strategy = strategy_module.FreezeTabular()
    cfg = _FakeTabularConfig(marker="direct")

    out = strategy.freeze(
        cfg,
        timestamp="2026-03-05T00:00:00",
        snapshot_id="s1",
        start_time=0.0,
        owner="tests",
    )

    assert out.snapshot_path == Path("snapshot/path")
    assert out.metadata == {"ok": True}


def test_freeze_tabular_validates_non_tabular_config_before_running(strategy_module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate raw-like config payload through registry before pipeline run."""
    calls: dict[str, Any] = {}

    def _validate(raw: dict, strategy_type: str) -> _FakeTabularConfig:
        calls["raw"] = raw
        calls["strategy_type"] = strategy_type
        return _FakeTabularConfig(marker="validated")

    monkeypatch.setattr(strategy_module, "validate_feature_registry", _validate)

    class _RawCfg:
        def dict(self) -> dict:
            return {"type": "tabular", "x": 1}

    strategy = strategy_module.FreezeTabular()
    strategy.freeze(
        _RawCfg(),
        timestamp="2026-03-05T00:00:00",
        snapshot_id="s1",
        start_time=0.0,
        owner="tests",
    )

    assert calls["raw"] == {"type": "tabular", "x": 1}
    assert calls["strategy_type"] == "tabular"
