"""Unit tests for CatBoost model-specific parameter preparation helpers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.exceptions import UserError

pytestmark = pytest.mark.unit


class _Dumpable:
    """Small config stub exposing pydantic-like ``model_dump`` behavior."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self, *, exclude_none: bool) -> dict[str, Any]:
        """Return payload with optional ``None`` filtering."""
        if not exclude_none:
            return dict(self.payload)
        return {k: v for k, v in self.payload.items() if v is not None}


def _import_module_with_catalog_stubs() -> types.ModuleType:
    """Import model-specific helper module with isolated registry catalogs."""
    module_name = "ml.runners.training.utils.model_specific.catboost"
    catalogs_module_name = "ml.registries.catalogs"

    sys.modules.pop(module_name, None)

    fake_catalogs = types.ModuleType(catalogs_module_name)
    fake_catalogs.__dict__["MODEL_CLASS_REGISTRY"] = {}
    fake_catalogs.__dict__["REGRESSION_LOSS_FUNCTIONS"] = {
        "mae": "MAE",
        "rmse": "RMSE",
    }
    sys.modules[catalogs_module_name] = fake_catalogs

    return importlib.import_module(module_name)


def test_extract_catboost_params_merges_model_and_ensemble_with_ensemble_precedence() -> None:
    """Combine model and ensemble params while dropping ``None`` values."""
    module = _import_module_with_catalog_stubs()

    cfg = cast(
        Any,
        SimpleNamespace(
            training=SimpleNamespace(
                model=_Dumpable({"depth": 6, "learning_rate": None, "l2_leaf_reg": 3.0}),
                ensemble=_Dumpable({"bagging_temperature": 0.5, "depth": 8}),
            )
        ),
    )

    params = module.extract_catboost_params(cfg)

    assert params == {"depth": 8, "l2_leaf_reg": 3.0, "bagging_temperature": 0.5}


def test_extract_catboost_params_returns_empty_when_optional_sections_missing() -> None:
    """Return empty params map when model and ensemble blocks are absent."""
    module = _import_module_with_catalog_stubs()

    cfg = cast(
        Any,
        SimpleNamespace(training=SimpleNamespace(model=None, ensemble=None)),
    )

    assert module.extract_catboost_params(cfg) == {}


def test_prepare_model_raises_user_error_for_unsupported_fixed_regression_metric() -> None:
    """Reject unsupported fixed regression metric values with explicit ``UserError``."""
    module = _import_module_with_catalog_stubs()

    cfg = cast(
        Any,
        SimpleNamespace(
            training=SimpleNamespace(
                iterations=100,
                model=None,
                ensemble=None,
                hardware=SimpleNamespace(task_type=SimpleNamespace(value="CPU"), devices=None),
            ),
            verbose=False,
            seed=7,
            task=SimpleNamespace(type="regression"),
            class_weighting=SimpleNamespace(policy="off"),
            scoring=SimpleNamespace(policy="fixed", fixed_metric="mape"),
            model_class="CatBoostRegressor",
        ),
    )

    with pytest.raises(UserError, match="Unsupported fixed metric mape"):
        module.prepare_model(
            cfg,
            cat_features=[],
            class_weights={},
            failure_management_dir=Path("failure_management/exp/train/run"),
        )


def test_prepare_model_uses_default_rmse_for_regression_default_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set CatBoost loss function to RMSE under regression-default scoring policy."""
    module = _import_module_with_catalog_stubs()

    captured_kwargs: dict[str, Any] = {}

    class _RecordingRegressor:
        """Minimal model stub that records constructor kwargs."""

        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def get_params(self) -> dict[str, Any]:
            """Expose kwargs for downstream logging in ``prepare_model``."""
            return captured_kwargs

    monkeypatch.setitem(module.MODEL_CLASS_REGISTRY, "CatBoostRegressor", _RecordingRegressor)

    cfg = cast(
        Any,
        SimpleNamespace(
            training=SimpleNamespace(
                iterations=64,
                model=None,
                ensemble=None,
                hardware=SimpleNamespace(task_type=SimpleNamespace(value="CPU"), devices=None),
            ),
            verbose=True,
            seed=2026,
            task=SimpleNamespace(type="regression"),
            class_weighting=SimpleNamespace(policy="off"),
            scoring=SimpleNamespace(policy="regression_default", fixed_metric=None),
            model_class="CatBoostRegressor",
        ),
    )

    module.prepare_model(
        cfg,
        cat_features=["hotel"],
        class_weights={},
        failure_management_dir=Path("failure_management/exp_3/train/run_3"),
    )

    assert captured_kwargs["loss_function"] == "RMSE"
    assert captured_kwargs["iterations"] == 64
    assert captured_kwargs["task_type"] == "CPU"
