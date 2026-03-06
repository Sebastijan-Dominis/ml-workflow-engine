"""Unit tests for CatBoost narrow-search pipeline step behavior."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def test_narrow_search_step_before_logs_start_message(caplog: pytest.LogCaptureFixture) -> None:
    """Emit the documented start log line from `before` hook."""
    narrow_search_module = _import_narrow_search_module()

    with caplog.at_level("INFO", logger=narrow_search_module.__name__):
        narrow_search_module.NarrowSearchStep().before(SimpleNamespace())

    assert "Starting narrow search step." in caplog.text


def test_narrow_search_step_after_logs_completion_message(caplog: pytest.LogCaptureFixture) -> None:
    """Emit the documented completion log line from `after` hook."""
    narrow_search_module = _import_narrow_search_module()

    with caplog.at_level("INFO", logger=narrow_search_module.__name__):
        narrow_search_module.NarrowSearchStep().after(SimpleNamespace())

    assert "Completed narrow search step." in caplog.text


def test_narrow_search_step_forwards_expected_args_validates_params_and_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Forward expected inputs, validate generated values, and persist narrow outputs."""
    narrow_search_module = _import_narrow_search_module()
    param_cfg = SimpleNamespace(model_dump=lambda exclude_none=True: {"model": {"depth": {}}})
    ctx = _make_context(tmp_path, narrow_enabled=True, narrow_param_cfg=param_cfg)

    pipeline_obj = object()
    model_obj = object()
    captured_prepare_kwargs: dict[str, Any] = {}
    validated_calls: list[tuple[str, object, str]] = []
    captured_randomized_kwargs: dict[str, Any] = {}
    captured_save_kwargs: dict[str, Any] = {}

    monkeypatch.setattr(narrow_search_module, "load_json", lambda _path, strict=False: {})

    def _prepare_narrow_params(**kwargs):
        captured_prepare_kwargs.update(kwargs)
        return {"Model__depth": [6, 7], "Model__l2_leaf_reg": [2.0]}

    monkeypatch.setattr(narrow_search_module, "prepare_narrow_params", _prepare_narrow_params)
    monkeypatch.setattr(
        narrow_search_module,
        "validate_param_value",
        lambda param, value, task_type: validated_calls.append((param, value, task_type)),
    )
    monkeypatch.setattr(narrow_search_module, "prepare_model", lambda *args, **kwargs: model_obj)
    monkeypatch.setattr(narrow_search_module, "build_pipeline_with_model", lambda **kwargs: pipeline_obj)

    def _perform_randomized_search(pipeline, **kwargs):
        captured_randomized_kwargs["pipeline"] = pipeline
        captured_randomized_kwargs.update(kwargs)
        return {
            "best_params": {"Model__depth": 7},
            "best_score": 0.82,
            "best_index": 1,
            "cv_results": {
                "mean_test_score": [0.8, 0.82],
                "std_test_score": [0.02, 0.01],
                "rank_test_score": [2, 1],
            },
            "param_distributions": {"Model__depth": [6, 7]},
            "n_iter": 2,
            "cv": 3,
            "scoring": "roc_auc",
            "random_state": 42,
            "error_score": "nan",
            "search_phase": "narrow",
        }

    monkeypatch.setattr(narrow_search_module, "perform_randomized_search", _perform_randomized_search)

    def _save_narrow(**kwargs):
        captured_save_kwargs.update(kwargs)

    monkeypatch.setattr(narrow_search_module, "save_narrow", _save_narrow)

    narrow_search_module.NarrowSearchStep().run(ctx)

    assert captured_prepare_kwargs["best_params"] == {"Model__depth": 6}
    assert captured_prepare_kwargs["narrow_params_cfg"] is param_cfg
    assert captured_prepare_kwargs["task_type"] == "CPU"

    assert validated_calls == [
        ("depth", 6, "CPU"),
        ("depth", 7, "CPU"),
        ("l2_leaf_reg", 2.0, "CPU"),
    ]

    assert captured_randomized_kwargs["pipeline"] is pipeline_obj
    assert captured_randomized_kwargs["X_train"].equals(ctx.require_x_train)
    assert captured_randomized_kwargs["y_train"].equals(ctx.require_y_train)
    assert captured_randomized_kwargs["param_distributions"] == {
        "Model__depth": [6, 7],
        "Model__l2_leaf_reg": [2.0],
    }
    assert captured_randomized_kwargs["model_cfg"] is ctx.model_cfg
    assert captured_randomized_kwargs["scoring"] == "roc_auc"
    assert captured_randomized_kwargs["search_phase"] == "narrow"

    assert ctx.best_params == {"Model__depth": 7}
    assert ctx.narrow_result["best_score"] == 0.82
    assert captured_save_kwargs["best_params"] == {"Model__depth": 7}
    assert captured_save_kwargs["tgt_file"] == tmp_path / "narrow_info.json"
