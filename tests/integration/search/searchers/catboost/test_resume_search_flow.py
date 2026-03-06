"""Integration tests for broad/narrow search resume behavior."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.search.searchers.catboost.pipeline.context import SearchContext

pytestmark = pytest.mark.integration


class _BroadParamDistributions:
    """Minimal broad-search config stub used by integration resume tests."""

    def model_dump(self, *, exclude_none: bool) -> dict[str, Any]:
        """Expose non-empty shape so broad search is considered configured."""
        return {"model": {"depth": [4, 6]}}

    def to_flat_dict(self) -> dict[str, list[int]]:
        """Return flattened broad parameter distributions."""
        return {"Model__depth": [4, 6]}


class _NarrowParamConfigurations:
    """Minimal narrow-search config stub used by integration resume tests."""

    def model_dump(self, *, exclude_none: bool) -> dict[str, Any]:
        """Expose non-empty shape so narrow search is considered configured."""
        return {"model": {"depth": {"step": 1}}}


def _import_step_modules_with_stubbed_heavy_deps() -> tuple[types.ModuleType, types.ModuleType]:
    """Import broad/narrow step modules with isolated heavy model-building deps."""
    broad_module_name = "ml.search.searchers.catboost.pipeline.steps.broad_search"
    narrow_module_name = "ml.search.searchers.catboost.pipeline.steps.narrow_search"
    build_module_name = "ml.modeling.catboost.build_pipeline_with_model"
    model_module_name = "ml.search.searchers.catboost.model"

    sys.modules.pop(broad_module_name, None)
    sys.modules.pop(narrow_module_name, None)
    original_build_module = sys.modules.get(build_module_name)
    original_model_module = sys.modules.get(model_module_name)

    fake_build_module = types.ModuleType(build_module_name)
    fake_build_module.__dict__["build_pipeline_with_model"] = lambda **kwargs: object()
    sys.modules[build_module_name] = fake_build_module

    fake_model_module = types.ModuleType(model_module_name)
    fake_model_module.__dict__["prepare_model"] = lambda *args, **kwargs: object()
    sys.modules[model_module_name] = fake_model_module

    try:
        broad_module = importlib.import_module(broad_module_name)
        narrow_module = importlib.import_module(narrow_module_name)
        return broad_module, narrow_module
    finally:
        if original_build_module is None:
            sys.modules.pop(build_module_name, None)
        else:
            sys.modules[build_module_name] = original_build_module

        if original_model_module is None:
            sys.modules.pop(model_module_name, None)
        else:
            sys.modules[model_module_name] = original_model_module


def _build_context(failure_management_dir: Path) -> SearchContext:
    """Build a search context with prepared-state fields needed by broad/narrow steps."""
    model_cfg = SimpleNamespace(
        problem="adr",
        version="v1",
        segment=SimpleNamespace(name="global"),
        search=SimpleNamespace(
            broad=SimpleNamespace(param_distributions=_BroadParamDistributions()),
            narrow=SimpleNamespace(enabled=True, param_configurations=_NarrowParamConfigurations()),
            hardware=SimpleNamespace(task_type=SimpleNamespace(value="CPU")),
        ),
    )

    ctx = SearchContext(
        model_cfg=cast(Any, model_cfg),
        strict=True,
        failure_management_dir=failure_management_dir,
    )
    ctx.X_train = pd.DataFrame({"country": ["PT", "GB"]})
    ctx.y_train = pd.Series([1, 0])
    ctx.input_schema = pd.DataFrame({"feature": ["country"], "dtype": ["object"]})
    ctx.derived_schema = pd.DataFrame({"feature": [], "source_operator": []})
    ctx.pipeline_cfg = {"steps": []}
    ctx.cat_features = ["country"]
    ctx.class_weights = {"class_weights": [1.0, 1.0]}
    ctx.scoring = "roc_auc"
    return ctx


def test_resume_flow_reuses_persisted_broad_and_narrow_results_without_recompute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist broad/narrow artifacts once, then validate second run resumes both phases."""
    broad_module, narrow_module = _import_step_modules_with_stubbed_heavy_deps()

    first_ctx = _build_context(tmp_path)

    monkeypatch.setattr(broad_module, "prepare_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(broad_module, "build_pipeline_with_model", lambda **kwargs: object())
    monkeypatch.setattr(
        broad_module,
        "perform_randomized_search",
        lambda *args, **kwargs: {
            "best_params": {"Model__depth": 6},
            "best_score": 0.81,
            "search_phase": "broad",
        },
    )

    monkeypatch.setattr(
        narrow_module,
        "prepare_narrow_params",
        lambda **kwargs: {"Model__depth": [6, 7]},
    )
    monkeypatch.setattr(narrow_module, "validate_param_value", lambda *args, **kwargs: None)
    monkeypatch.setattr(narrow_module, "prepare_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(narrow_module, "build_pipeline_with_model", lambda **kwargs: object())
    monkeypatch.setattr(
        narrow_module,
        "perform_randomized_search",
        lambda *args, **kwargs: {
            "best_params": {"Model__depth": 7},
            "best_score": 0.83,
            "search_phase": "narrow",
        },
    )

    broad_module.BroadSearchStep().run(first_ctx)
    narrow_module.NarrowSearchStep().run(first_ctx)

    assert first_ctx.best_params_1 == {"Model__depth": 6}
    assert first_ctx.best_params == {"Model__depth": 7}
    assert (tmp_path / "broad_info.json").exists()
    assert (tmp_path / "narrow_info.json").exists()

    resumed_ctx = _build_context(tmp_path)

    monkeypatch.setattr(
        broad_module,
        "perform_randomized_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("broad search should not rerun")),
    )
    monkeypatch.setattr(
        narrow_module,
        "perform_randomized_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("narrow search should not rerun")),
    )

    broad_module.BroadSearchStep().run(resumed_ctx)
    narrow_module.NarrowSearchStep().run(resumed_ctx)

    assert resumed_ctx.best_params_1 == {"Model__depth": 6}
    assert resumed_ctx.best_params == {"Model__depth": 7}
    assert resumed_ctx.require_broad_result["best_score"] == 0.81
    assert resumed_ctx.require_narrow_result["best_score"] == 0.83
    assert resumed_ctx.narrow_disabled is False
