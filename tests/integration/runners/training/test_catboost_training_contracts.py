"""Integration tests for training contracts across multiple CatBoost components."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytestmark = pytest.mark.integration


def _import_model_specific_module() -> types.ModuleType:
    """Import model-specific CatBoost helpers with isolated registry catalogs."""
    module_name = "ml.runners.training.utils.model_specific.catboost"
    catalogs_module_name = "ml.registries.catalogs"

    sys.modules.pop(module_name, None)

    fake_catalogs = types.ModuleType(catalogs_module_name)
    fake_catalogs.__dict__["MODEL_CLASS_REGISTRY"] = {}
    fake_catalogs.__dict__["REGRESSION_LOSS_FUNCTIONS"] = {
        "mae": "MAE",
        "mse": "RMSE",
        "rmse": "RMSE",
    }
    sys.modules[catalogs_module_name] = fake_catalogs

    return importlib.import_module(module_name)


def test_prepare_model_sets_gpu_devices_and_regression_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a regressor with GPU settings and fixed-metric loss translation."""
    module = _import_model_specific_module()

    captured_kwargs: dict[str, Any] = {}

    class _RecordingRegressor:
        """Record constructor kwargs while exposing CatBoost-like params API."""

        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def get_params(self) -> dict[str, Any]:
            """Return captured kwargs to satisfy trainer logging contract."""
            return captured_kwargs

    monkeypatch.setitem(module.MODEL_CLASS_REGISTRY, "CatBoostRegressor", _RecordingRegressor)

    cfg = cast(
        Any,
        SimpleNamespace(
            training=SimpleNamespace(
                iterations=150,
                model=None,
                ensemble=None,
                hardware=SimpleNamespace(
                    task_type=SimpleNamespace(value="GPU"),
                    devices="0",
                ),
            ),
            verbose=False,
            seed=123,
            task=SimpleNamespace(type="regression"),
            class_weighting=SimpleNamespace(policy="off"),
            scoring=SimpleNamespace(policy="fixed", fixed_metric="mae"),
            model_class="CatBoostRegressor",
        ),
    )

    model = module.prepare_model(
        cfg,
        cat_features=["hotel", "agent"],
        class_weights={},
        failure_management_dir=Path("failure_management/exp_1/train/run_1"),
    )

    assert isinstance(model, _RecordingRegressor)
    assert captured_kwargs["iterations"] == 150
    assert captured_kwargs["task_type"] == "GPU"
    assert captured_kwargs["devices"] == "0"
    assert captured_kwargs["loss_function"] == "MAE"
    assert captured_kwargs["cat_features"] == ["hotel", "agent"]
    assert captured_kwargs["train_dir"].endswith("run_1")


def test_prepare_model_includes_class_weights_for_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attach computed class weights only for classification policy-enabled runs."""
    module = _import_model_specific_module()

    captured_kwargs: dict[str, Any] = {}

    class _RecordingClassifier:
        """Minimal classifier stand-in capturing constructor payload."""

        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def get_params(self) -> dict[str, Any]:
            """Expose params map required by downstream logging."""
            return captured_kwargs

    monkeypatch.setitem(module.MODEL_CLASS_REGISTRY, "CatBoostClassifier", _RecordingClassifier)

    cfg = cast(
        Any,
        SimpleNamespace(
            training=SimpleNamespace(
                iterations=80,
                model=None,
                ensemble=None,
                hardware=SimpleNamespace(task_type=SimpleNamespace(value="CPU"), devices=None),
            ),
            verbose=True,
            seed=42,
            task=SimpleNamespace(type="classification"),
            class_weighting=SimpleNamespace(policy="if_imbalanced"),
            scoring=SimpleNamespace(policy="classification_default", fixed_metric=None),
            model_class="CatBoostClassifier",
        ),
    )

    class_weights = {"class_weights": [1.0, 2.5]}

    module.prepare_model(
        cfg,
        cat_features=["segment"],
        class_weights=class_weights,
        failure_management_dir=Path("failure_management/exp_2/train/run_2"),
    )

    assert captured_kwargs["class_weights"] == [1.0, 2.5]
    assert "loss_function" not in captured_kwargs
    assert "devices" not in captured_kwargs
