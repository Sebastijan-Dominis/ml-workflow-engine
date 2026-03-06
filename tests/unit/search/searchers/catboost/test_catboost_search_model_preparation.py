"""Unit tests for CatBoost search-phase model preparation helper."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytestmark = pytest.mark.unit


def _import_search_model_module() -> types.ModuleType:
    """Import search model helper module with isolated registry catalogs."""
    module_name = "ml.search.searchers.catboost.model"
    catalogs_module_name = "ml.registries.catalogs"

    sys.modules.pop(module_name, None)
    original_catalogs = sys.modules.get(catalogs_module_name)

    fake_catalogs = types.ModuleType(catalogs_module_name)
    fake_catalogs.__dict__["MODEL_CLASS_REGISTRY"] = {}
    sys.modules[catalogs_module_name] = fake_catalogs

    try:
        return importlib.import_module(module_name)
    finally:
        if original_catalogs is None:
            sys.modules.pop(catalogs_module_name, None)
        else:
            sys.modules[catalogs_module_name] = original_catalogs


def test_prepare_model_adds_class_weights_for_classifier_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attach class_weights only for classifier runs with non-off weighting policy."""
    module = _import_search_model_module()

    captured_kwargs: dict[str, Any] = {}

    class _RecordingClassifier:
        """Record kwargs and expose ``get_params`` for logging calls."""

        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def get_params(self) -> dict[str, Any]:
            """Return constructor kwargs to emulate CatBoost API."""
            return captured_kwargs

    monkeypatch.setitem(module.MODEL_CLASS_REGISTRY, "classifier", _RecordingClassifier)

    cfg = cast(
        Any,
        SimpleNamespace(
            model_class="classifier",
            verbose=False,
            seed=123,
            class_weighting=SimpleNamespace(policy="if_imbalanced"),
            search=SimpleNamespace(
                hardware=SimpleNamespace(task_type=SimpleNamespace(value="CPU"), devices="0"),
                broad=SimpleNamespace(iterations=80),
                narrow=SimpleNamespace(iterations=40),
            ),
        ),
    )

    model = module.prepare_model(
        cfg,
        search_phase="broad",
        cat_features=["country"],
        class_weights={"class_weights": [1.0, 2.5]},
    )

    assert isinstance(model, _RecordingClassifier)
    assert captured_kwargs["iterations"] == 80
    assert captured_kwargs["task_type"] == "CPU"
    assert captured_kwargs["devices"] == "0"
    assert captured_kwargs["cat_features"] == ["country"]
    assert captured_kwargs["class_weights"] == [1.0, 2.5]


def test_prepare_model_omits_class_weights_for_regressor_or_off_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not pass class_weights unless classifier + non-off policy + payload are present."""
    module = _import_search_model_module()

    captured_kwargs: dict[str, Any] = {}

    class _RecordingRegressor:
        """Record kwargs and expose ``get_params`` for logging calls."""

        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        def get_params(self) -> dict[str, Any]:
            """Return constructor kwargs to emulate CatBoost API."""
            return captured_kwargs

    monkeypatch.setitem(module.MODEL_CLASS_REGISTRY, "regressor", _RecordingRegressor)

    cfg = cast(
        Any,
        SimpleNamespace(
            model_class="regressor",
            verbose=True,
            seed=9,
            class_weighting=SimpleNamespace(policy="off"),
            search=SimpleNamespace(
                hardware=SimpleNamespace(task_type=SimpleNamespace(value="GPU"), devices="0:1"),
                broad=SimpleNamespace(iterations=100),
                narrow=SimpleNamespace(iterations=50),
            ),
        ),
    )

    model = module.prepare_model(
        cfg,
        search_phase="narrow",
        cat_features=[],
        class_weights={"class_weights": [1.0, 2.0]},
    )

    assert isinstance(model, _RecordingRegressor)
    assert captured_kwargs["iterations"] == 50
    assert captured_kwargs["task_type"] == "GPU"
    assert captured_kwargs["devices"] == "0:1"
    assert "class_weights" not in captured_kwargs
