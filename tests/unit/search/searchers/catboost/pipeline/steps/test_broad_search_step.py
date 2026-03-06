"""Unit tests for CatBoost broad-search pipeline step behavior."""

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


def _import_broad_search_module() -> types.ModuleType:
    """Import broad-search step module with isolated stub dependencies."""
    module_name = "ml.search.searchers.catboost.pipeline.steps.broad_search"
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


def _make_context(tmp_path: Path) -> SimpleNamespace:
    """Create a minimal broad-search context stub with required fields."""
    return SimpleNamespace(
        failure_management_dir=tmp_path,
        model_cfg=SimpleNamespace(
            problem="adr",
            version="v1",
            segment=SimpleNamespace(name="global"),
            search=SimpleNamespace(
                broad=SimpleNamespace(param_distributions=None),
            ),
        ),
        class_weights=None,
        require_cat_features=["country"],
        require_pipeline_cfg={"steps": []},
        require_input_schema=pd.DataFrame({"feature": ["country"], "dtype": ["object"]}),
        require_derived_schema=pd.DataFrame({"feature": [], "source_operator": []}),
        require_x_train=pd.DataFrame({"country": ["PT"]}),
        require_y_train=pd.Series([1]),
        require_scoring="roc_auc",
    )


def test_broad_search_step_uses_existing_broad_info_and_skips_compute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reuse persisted broad-search state and avoid model/pipeline/search execution."""
    broad_search_module = _import_broad_search_module()
    ctx = _make_context(tmp_path)

    monkeypatch.setattr(
        broad_search_module,
        "load_json",
        lambda _path, strict=False: {
            "broad_result": {"best_params": {"Model__depth": 6}},
            "best_params_1": {"Model__depth": 6},
        },
    )
    monkeypatch.setattr(
        broad_search_module,
        "prepare_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    result = broad_search_module.BroadSearchStep().run(ctx)

    assert result is ctx
    assert ctx.broad_result == {"best_params": {"Model__depth": 6}}
    assert ctx.best_params_1 == {"Model__depth": 6}


def test_broad_search_step_raises_user_error_for_malformed_resume_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `UserError` when persisted broad info misses required keys."""
    broad_search_module = _import_broad_search_module()
    ctx = _make_context(tmp_path)

    monkeypatch.setattr(
        broad_search_module,
        "load_json",
        lambda _path, strict=False: {"broad_result": {"best_params": {"Model__depth": 6}}},
    )

    with pytest.raises(UserError, match="missing required keys"):
        broad_search_module.BroadSearchStep().run(ctx)


def test_broad_search_step_raises_config_error_when_param_distributions_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `ConfigError` when broad param distributions are undefined or empty."""
    broad_search_module = _import_broad_search_module()
    ctx = _make_context(tmp_path)
    ctx.model_cfg.search.broad.param_distributions = SimpleNamespace(
        model_dump=lambda exclude_none=True: {},
        to_flat_dict=lambda: {},
    )

    monkeypatch.setattr(
        broad_search_module,
        "load_json",
        lambda _path, strict=False: {},
    )
    monkeypatch.setattr(broad_search_module, "prepare_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(broad_search_module, "build_pipeline_with_model", lambda **kwargs: object())

    with pytest.raises(ConfigError, match="No broad search param_distributions defined"):
        broad_search_module.BroadSearchStep().run(ctx)


def test_broad_search_step_wraps_randomized_search_failures_as_search_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `SearchError` with context when underlying broad randomized search fails."""
    broad_search_module = _import_broad_search_module()
    ctx = _make_context(tmp_path)
    ctx.model_cfg.search.broad.param_distributions = SimpleNamespace(
        model_dump=lambda exclude_none=True: {"model": {"depth": [4, 6]}},
        to_flat_dict=lambda: {"Model__depth": [4, 6]},
    )

    monkeypatch.setattr(broad_search_module, "load_json", lambda _path, strict=False: {})
    monkeypatch.setattr(broad_search_module, "prepare_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(broad_search_module, "build_pipeline_with_model", lambda **kwargs: object())
    monkeypatch.setattr(
        broad_search_module,
        "perform_randomized_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("search failed")),
    )

    with pytest.raises(SearchError, match="Broad hyperparameter search failed"):
        broad_search_module.BroadSearchStep().run(ctx)


def test_broad_search_step_before_logs_start_message(caplog: pytest.LogCaptureFixture) -> None:
    """Emit the documented start log line from `before` hook."""
    broad_search_module = _import_broad_search_module()

    with caplog.at_level("INFO", logger=broad_search_module.__name__):
        broad_search_module.BroadSearchStep().before(SimpleNamespace())

    assert "Starting broad search step." in caplog.text


def test_broad_search_step_after_logs_completion_message(caplog: pytest.LogCaptureFixture) -> None:
    """Emit the documented completion log line from `after` hook."""
    broad_search_module = _import_broad_search_module()

    with caplog.at_level("INFO", logger=broad_search_module.__name__):
        broad_search_module.BroadSearchStep().after(SimpleNamespace())

    assert "Completed broad search step." in caplog.text


def test_broad_search_step_forwards_expected_args_and_persists_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Forward expected arguments into search helper and persist broad outputs."""
    broad_search_module = _import_broad_search_module()
    ctx = _make_context(tmp_path)
    ctx.model_cfg.search.broad.param_distributions = SimpleNamespace(
        model_dump=lambda exclude_none=True: {"model": {"depth": [4, 6]}},
        to_flat_dict=lambda: {"Model__depth": [4, 6]},
    )

    pipeline_obj = object()
    model_obj = object()
    captured_randomized_kwargs: dict[str, Any] = {}
    captured_save_kwargs: dict[str, Any] = {}

    monkeypatch.setattr(broad_search_module, "load_json", lambda _path, strict=False: {})
    monkeypatch.setattr(broad_search_module, "prepare_model", lambda *args, **kwargs: model_obj)
    monkeypatch.setattr(
        broad_search_module,
        "build_pipeline_with_model",
        lambda **kwargs: pipeline_obj,
    )

    def _perform_randomized_search(pipeline, **kwargs):
        captured_randomized_kwargs["pipeline"] = pipeline
        captured_randomized_kwargs.update(kwargs)
        return {
            "best_params": {"Model__depth": 6},
            "best_score": 0.8,
            "best_index": 0,
            "cv_results": {
                "mean_test_score": [0.8],
                "std_test_score": [0.01],
                "rank_test_score": [1],
            },
            "param_distributions": {"Model__depth": [4, 6]},
            "n_iter": 2,
            "cv": 3,
            "scoring": "roc_auc",
            "random_state": 42,
            "error_score": "nan",
            "search_phase": "broad",
        }

    monkeypatch.setattr(broad_search_module, "perform_randomized_search", _perform_randomized_search)

    def _save_broad(**kwargs):
        captured_save_kwargs.update(kwargs)

    monkeypatch.setattr(broad_search_module, "save_broad", _save_broad)

    broad_search_module.BroadSearchStep().run(ctx)

    assert captured_randomized_kwargs["pipeline"] is pipeline_obj
    assert captured_randomized_kwargs["X_train"].equals(ctx.require_x_train)
    assert captured_randomized_kwargs["y_train"].equals(ctx.require_y_train)
    assert captured_randomized_kwargs["param_distributions"] == {"Model__depth": [4, 6]}
    assert captured_randomized_kwargs["model_cfg"] is ctx.model_cfg
    assert captured_randomized_kwargs["scoring"] == "roc_auc"
    assert captured_randomized_kwargs["search_phase"] == "broad"

    assert ctx.best_params_1 == {"Model__depth": 6}
    assert ctx.broad_result["best_score"] == 0.8
    assert captured_save_kwargs["best_params_1"] == {"Model__depth": 6}
    assert captured_save_kwargs["tgt_file"] == tmp_path / "broad_info.json"
