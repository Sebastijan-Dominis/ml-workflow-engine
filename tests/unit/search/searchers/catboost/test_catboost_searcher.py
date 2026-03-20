"""Unit tests for CatBoost searcher orchestration and output mapping."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _import_searcher_module_with_stubs() -> types.ModuleType:
    """Import CatBoost searcher with isolated step/context/runner dependencies."""
    module_name = "ml.search.searchers.catboost.catboost"
    context_module_name = "ml.search.searchers.catboost.pipeline.context"
    prep_module_name = "ml.search.searchers.catboost.pipeline.steps.preparation"
    broad_module_name = "ml.search.searchers.catboost.pipeline.steps.broad_search"
    narrow_module_name = "ml.search.searchers.catboost.pipeline.steps.narrow_search"
    creator_module_name = "ml.search.searchers.catboost.search_results_creator"
    runner_module_name = "ml.utils.pipeline_core.runner"

    sys.modules.pop(module_name, None)

    stub_module_names = [
        context_module_name,
        prep_module_name,
        broad_module_name,
        narrow_module_name,
        creator_module_name,
        runner_module_name,
    ]
    original_modules = {name: sys.modules.get(name) for name in stub_module_names}

    fake_context_module = types.ModuleType(context_module_name)

    class _FakeContext:
        def __init__(self, *, model_cfg: Any, strict: bool, failure_management_dir: Path, snapshot_binding_key: str | None = None) -> None:
            self.model_cfg = model_cfg
            self.strict = strict
            self.failure_management_dir = failure_management_dir
            self.snapshot_binding_key = snapshot_binding_key
            self.feature_lineage = []
            self.pipeline_hash = ""
            self.scoring = ""
            self.splits_info = {}

        @property
        def require_feature_lineage(self) -> list[Any]:
            return self.feature_lineage

        @property
        def require_pipeline_hash(self) -> str:
            return self.pipeline_hash

        @property
        def require_scoring(self) -> str:
            return self.scoring

        @property
        def require_splits_info(self) -> dict[str, Any]:
            return self.splits_info

    fake_context_module.__dict__["SearchContext"] = _FakeContext
    sys.modules[context_module_name] = fake_context_module

    fake_prep_module = types.ModuleType(prep_module_name)

    class _PreparationStep:
        pass

    fake_prep_module.__dict__["PreparationStep"] = _PreparationStep
    sys.modules[prep_module_name] = fake_prep_module

    fake_broad_module = types.ModuleType(broad_module_name)

    class _BroadSearchStep:
        pass

    fake_broad_module.__dict__["BroadSearchStep"] = _BroadSearchStep
    sys.modules[broad_module_name] = fake_broad_module

    fake_narrow_module = types.ModuleType(narrow_module_name)

    class _NarrowSearchStep:
        pass

    fake_narrow_module.__dict__["NarrowSearchStep"] = _NarrowSearchStep
    sys.modules[narrow_module_name] = fake_narrow_module

    fake_creator_module = types.ModuleType(creator_module_name)
    fake_creator_module.__dict__["create_search_results"] = lambda ctx: {"placeholder": True}
    sys.modules[creator_module_name] = fake_creator_module

    fake_runner_module = types.ModuleType(runner_module_name)

    class _PipelineRunner:
        def __init__(self, steps: list[Any]) -> None:
            self.steps = steps

        def run(self, ctx: Any) -> Any:
            return ctx

    fake_runner_module.__dict__["PipelineRunner"] = _PipelineRunner
    sys.modules[runner_module_name] = fake_runner_module

    try:
        return importlib.import_module(module_name)
    finally:
        for name in stub_module_names:
            original_module = original_modules[name]
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_catboost_searcher_wires_pipeline_steps_and_maps_context_to_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Construct runner with expected step order and expose context-derived output fields."""
    module = _import_searcher_module_with_stubs()

    captured_runner_init: dict[str, Any] = {}

    class _RecordingRunner:
        def __init__(self, steps: list[Any]) -> None:
            captured_runner_init["steps"] = steps

        def run(self, ctx: Any) -> Any:
            ctx.feature_lineage = [SimpleNamespace(feature="country")]
            ctx.pipeline_hash = "pipeline-hash-123"
            ctx.scoring = "roc_auc"
            ctx.splits_info = {"train_rows": 100, "val_rows": 25}
            return ctx

    monkeypatch.setattr(module, "PipelineRunner", _RecordingRunner)

    captured_creator_ctx: dict[str, Any] = {}

    def _create_search_results(ctx: Any) -> dict[str, Any]:
        captured_creator_ctx["ctx"] = ctx
        return {"best_pipeline_params": {"Model__depth": 6}}

    monkeypatch.setattr(module, "create_search_results", _create_search_results)

    cfg = SimpleNamespace(problem="adr", segment=SimpleNamespace(name="global"), version="v1")
    searcher = module.CatBoostSearcher()

    output = searcher.search(
        cfg,
        strict=False,
        failure_management_dir=tmp_path / "failure_management" / "exp_7",
    )

    step_names = [type(step).__name__ for step in captured_runner_init["steps"]]
    assert step_names == ["_PreparationStep", "_BroadSearchStep", "_NarrowSearchStep"]

    assert captured_creator_ctx["ctx"].model_cfg is cfg
    assert captured_creator_ctx["ctx"].strict is False

    assert output.search_results == {"best_pipeline_params": {"Model__depth": 6}}
    assert output.pipeline_hash == "pipeline-hash-123"
    assert output.scoring_method == "roc_auc"
    assert output.splits_info == {"train_rows": 100, "val_rows": 25}
    assert output.feature_lineage[0].feature == "country"
