"""Unit tests for low-level CatBoost training helper behavior."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from ml.exceptions import TrainingError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pytestmark = pytest.mark.unit


def _import_module_with_stubbed_composition() -> types.ModuleType:
    """Import module under test with isolated composition dependency."""
    module_name = "ml.runners.training.trainers.catboost.train_catboost_model"
    composition_module_name = "ml.pipelines.composition.add_model_to_pipeline"

    sys.modules.pop(module_name, None)
    original_composition = sys.modules.get(composition_module_name)

    fake_composition = types.ModuleType(composition_module_name)
    fake_composition.__dict__["add_model_to_pipeline"] = (
        lambda pipeline, model: Pipeline(pipeline.steps + [("Model", model)])
    )
    sys.modules[composition_module_name] = fake_composition

    try:
        return importlib.import_module(module_name)
    finally:
        if original_composition is None:
            sys.modules.pop(composition_module_name, None)
        else:
            sys.modules[composition_module_name] = original_composition


class _RecordingModel:
    """Minimal estimator stub that records fit invocations for assertions."""

    def __init__(self) -> None:
        self.fit_calls: list[dict[str, Any]] = []

    def fit(self, X: Any, y: Any, **kwargs: Any) -> _RecordingModel:
        """Store incoming fit payload and mimic sklearn-style fluent fit."""
        self.fit_calls.append({"X": X, "y": y, "kwargs": kwargs})
        return self


class _ExplodingModel:
    """Estimator stub that raises during fit to exercise wrapped failure paths."""

    def fit(self, _X: Any, _y: Any, **_kwargs: Any) -> None:
        """Always fail to validate error wrapping and context in messages."""
        raise RuntimeError("boom")


def test_train_catboost_model_uses_snapshot_and_no_eval_set_when_early_stopping_disabled() -> None:
    """Fit with snapshot kwargs only when early stopping rounds are unset."""
    module = _import_module_with_stubbed_composition()

    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.0, 1.0, 0.0]})
    y_train = pd.Series([0, 1, 0])
    X_val = pd.DataFrame({"f1": [4.0, 5.0], "f2": [1.0, 1.0]})
    y_val = pd.Series([1, 0])

    model = _RecordingModel()
    steps = [("scale", StandardScaler()), ("Model", model)]

    cfg = cast(
        Any,
        SimpleNamespace(
            problem="no_show",
            segment=SimpleNamespace(name="global"),
            version="v1",
            training=SimpleNamespace(
                snapshot_interval_seconds=60,
                early_stopping_rounds=None,
            ),
        ),
    )

    trained_model, pipeline = module.train_catboost_model(
        cfg,
        steps=steps,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    assert trained_model is model
    assert isinstance(pipeline, Pipeline)
    assert pipeline.steps[-1][0] == "Model"
    assert pipeline.steps[-1][1] is model

    fit_kwargs = model.fit_calls[0]["kwargs"]
    assert fit_kwargs["use_best_model"] is True
    assert fit_kwargs["save_snapshot"] is True
    assert fit_kwargs["snapshot_file"] == "catboost_snapshot.bin"
    assert fit_kwargs["snapshot_interval"] == 60
    assert "eval_set" not in fit_kwargs
    assert "early_stopping_rounds" not in fit_kwargs


def test_train_catboost_model_adds_eval_set_when_early_stopping_enabled() -> None:
    """Transform validation features and pass eval_set + early stopping kwargs."""
    module = _import_module_with_stubbed_composition()

    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.0, 1.0, 0.0]})
    y_train = pd.Series([0, 1, 0])
    X_val = pd.DataFrame({"f1": [4.0, 5.0], "f2": [1.0, 1.0]})
    y_val = pd.Series([1, 0])

    model = _RecordingModel()
    steps = [("scale", StandardScaler()), ("Model", model)]

    cfg = cast(
        Any,
        SimpleNamespace(
            problem="no_show",
            segment=SimpleNamespace(name="global"),
            version="v1",
            training=SimpleNamespace(
                snapshot_interval_seconds=20,
                early_stopping_rounds=15,
            ),
        ),
    )

    module.train_catboost_model(
        cfg,
        steps=steps,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    fit_payload = model.fit_calls[0]
    fit_kwargs = fit_payload["kwargs"]
    assert fit_kwargs["early_stopping_rounds"] == 15
    assert "eval_set" in fit_kwargs

    eval_X, eval_y = fit_kwargs["eval_set"]
    assert eval_y.equals(y_val)
    assert isinstance(eval_X, np.ndarray)
    assert eval_X.shape[0] == len(y_val)


def test_train_catboost_model_wraps_underlying_exceptions_with_training_context() -> None:
    """Raise ``TrainingError`` that includes model identity for easier triage."""
    module = _import_module_with_stubbed_composition()

    cfg = cast(
        Any,
        SimpleNamespace(
            problem="cancellation",
            segment=SimpleNamespace(name="city_hotel"),
            version="v2",
            training=SimpleNamespace(snapshot_interval_seconds=10, early_stopping_rounds=5),
        ),
    )

    with pytest.raises(
        TrainingError,
        match="cancellation_city_hotel_v2",
    ):
        module.train_catboost_model(
            cfg,
            steps=[("scale", StandardScaler()), ("Model", _ExplodingModel())],
            X_train=pd.DataFrame({"f1": [1.0, 2.0]}),
            y_train=pd.Series([0, 1]),
            X_val=pd.DataFrame({"f1": [3.0]}),
            y_val=pd.Series([1]),
        )
