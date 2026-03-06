"""Unit tests for CatBoost broad-search pipeline step behavior."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from ml.exceptions import ConfigError, UserError

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
