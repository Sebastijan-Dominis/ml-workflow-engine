"""Unit tests for CatBoost narrow-search pipeline step behavior."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from ml.exceptions import ConfigError, SearchError, UserError

pytestmark = pytest.mark.unit


def _import_narrow_search_module() -> types.ModuleType:
    """Import narrow-search step module with isolated heavy dependencies."""
    module_name = "ml.search.searchers.catboost.pipeline.steps.narrow_search"
    build_module_name = "ml.modeling.catboost.build_pipeline_with_model"
    model_module_name = "ml.search.searchers.catboost.model"

    sys.modules.pop(module_name, None)

    fake_build_module = types.ModuleType(build_module_name)
    fake_build_module.__dict__["build_pipeline_with_model"] = lambda **kwargs: object()
    sys.modules[build_module_name] = fake_build_module

    fake_model_module = types.ModuleType(model_module_name)
    fake_model_module.__dict__["prepare_model"] = lambda *args, **kwargs: object()
    sys.modules[model_module_name] = fake_model_module

    return importlib.import_module(module_name)


def _make_context(tmp_path: Path, *, narrow_enabled: bool, narrow_param_cfg: object) -> SimpleNamespace:
    """Create a minimal narrow-search context stub with required fields."""
    return SimpleNamespace(
        failure_management_dir=tmp_path,
        model_cfg=SimpleNamespace(
            problem="adr",
            version="v1",
            segment=SimpleNamespace(name="global"),
            search=SimpleNamespace(
                narrow=SimpleNamespace(enabled=narrow_enabled, param_configurations=narrow_param_cfg),
                hardware=SimpleNamespace(task_type=SimpleNamespace(value="CPU")),
            ),
        ),
        class_weights=None,
        require_best_params_1={"Model__depth": 6},
        require_cat_features=["country"],
        require_pipeline_cfg={"steps": []},
        require_input_schema=pd.DataFrame({"feature": ["country"], "dtype": ["object"]}),
        require_derived_schema=pd.DataFrame({"feature": [], "source_operator": []}),
        require_x_train=pd.DataFrame({"country": ["PT"]}),
        require_y_train=pd.Series([1]),
        require_scoring="roc_auc",
    )


def test_narrow_search_step_uses_existing_narrow_info_and_skips_compute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reuse persisted narrow-search state and avoid model/pipeline/search execution."""
    narrow_search_module = _import_narrow_search_module()
    ctx = _make_context(
        tmp_path,
        narrow_enabled=True,
        narrow_param_cfg=SimpleNamespace(model_dump=lambda exclude_none=True: {"model": {}}),
    )

    monkeypatch.setattr(
        narrow_search_module,
        "load_json",
        lambda _path, strict=False: {
            "narrow_result": {"best_params": {"Model__depth": 7}},
            "best_params": {"Model__depth": 7},
        },
    )
    monkeypatch.setattr(
        narrow_search_module,
        "prepare_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    result = narrow_search_module.NarrowSearchStep().run(ctx)

    assert result is ctx
    assert ctx.narrow_disabled is False
    assert ctx.narrow_result == {"best_params": {"Model__depth": 7}}
    assert ctx.best_params == {"Model__depth": 7}


def test_narrow_search_step_raises_user_error_for_malformed_resume_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `UserError` when persisted narrow info misses required keys."""
    narrow_search_module = _import_narrow_search_module()
    ctx = _make_context(
        tmp_path,
        narrow_enabled=True,
        narrow_param_cfg=SimpleNamespace(model_dump=lambda exclude_none=True: {"model": {}}),
    )

    monkeypatch.setattr(
        narrow_search_module,
        "load_json",
        lambda _path, strict=False: {"narrow_result": {"best_params": {"Model__depth": 7}}},
    )

    with pytest.raises(UserError, match="missing required keys"):
        narrow_search_module.NarrowSearchStep().run(ctx)


def test_narrow_search_step_sets_flag_and_returns_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Set `narrow_disabled=True` and return without further processing when disabled."""
    narrow_search_module = _import_narrow_search_module()
    ctx = _make_context(
        tmp_path,
        narrow_enabled=False,
        narrow_param_cfg=SimpleNamespace(model_dump=lambda exclude_none=True: {"model": {}}),
    )

    monkeypatch.setattr(narrow_search_module, "load_json", lambda _path, strict=False: {})

    result = narrow_search_module.NarrowSearchStep().run(ctx)

    assert result is ctx
    assert ctx.narrow_disabled is True


def test_narrow_search_step_raises_config_error_when_param_configurations_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `ConfigError` when narrow param configuration is undefined or empty."""
    narrow_search_module = _import_narrow_search_module()
    ctx = _make_context(
        tmp_path,
        narrow_enabled=True,
        narrow_param_cfg=SimpleNamespace(model_dump=lambda exclude_none=True: {}),
    )

    monkeypatch.setattr(narrow_search_module, "load_json", lambda _path, strict=False: {})

    with pytest.raises(ConfigError, match="No narrow search param_configurations defined"):
        narrow_search_module.NarrowSearchStep().run(ctx)


def test_narrow_search_step_wraps_randomized_search_failures_as_search_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `SearchError` with context when underlying narrow randomized search fails."""
    narrow_search_module = _import_narrow_search_module()
    ctx = _make_context(
        tmp_path,
        narrow_enabled=True,
        narrow_param_cfg=SimpleNamespace(model_dump=lambda exclude_none=True: {"model": {"depth": {}}}),
    )

    monkeypatch.setattr(narrow_search_module, "load_json", lambda _path, strict=False: {})
    monkeypatch.setattr(
        narrow_search_module,
        "prepare_narrow_params",
        lambda **kwargs: {"Model__depth": [6, 7]},
    )
    monkeypatch.setattr(narrow_search_module, "validate_param_value", lambda *args, **kwargs: None)
    monkeypatch.setattr(narrow_search_module, "prepare_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(narrow_search_module, "build_pipeline_with_model", lambda **kwargs: object())
    monkeypatch.setattr(
        narrow_search_module,
        "perform_randomized_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("search failed")),
    )

    with pytest.raises(SearchError, match="Narrow hyperparameter search failed"):
        narrow_search_module.NarrowSearchStep().run(ctx)
