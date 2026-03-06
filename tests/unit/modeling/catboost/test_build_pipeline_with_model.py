"""Unit tests for CatBoost pipeline-plus-model assembly helper."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor
from ml.exceptions import PipelineContractError
from sklearn.pipeline import Pipeline

pytestmark = pytest.mark.unit


def _dummy_model_cfg() -> Any:
    """Return a lightweight config stub matching attribute access expectations."""
    return cast(Any, SimpleNamespace())


def _import_build_module() -> types.ModuleType:
    """Import module under test with stubbed pipeline dependencies to avoid circular imports."""
    module_name = "ml.modeling.catboost.build_pipeline_with_model"
    builders_module_name = "ml.pipelines.builders"
    composition_module_name = "ml.pipelines.composition.add_model_to_pipeline"

    sys.modules.pop(module_name, None)

    fake_builders_module = types.ModuleType(builders_module_name)
    fake_builders_module.__dict__["build_pipeline"] = lambda **_: Pipeline(steps=[])
    sys.modules[builders_module_name] = fake_builders_module

    fake_composition_module = types.ModuleType(composition_module_name)
    fake_composition_module.__dict__["add_model_to_pipeline"] = lambda pipeline, _model: pipeline
    sys.modules[composition_module_name] = fake_composition_module

    return importlib.import_module(module_name)


def test_build_pipeline_with_model_wires_classifier_into_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a feature pipeline first, then append CatBoost classifier as final model step."""
    build_module = _import_build_module()
    model_cfg = _dummy_model_cfg()
    pipeline_cfg = {"steps": ["impute", "encode"]}
    input_schema = pd.DataFrame({"name": ["feature_a"], "dtype": ["float64"]})
    derived_schema = pd.DataFrame({"name": ["feature_b"], "dtype": ["float64"]})
    model = CatBoostClassifier(verbose=False)

    base_pipeline = Pipeline(steps=[])
    final_pipeline = Pipeline(steps=[("Model", model)])

    captured: dict[str, Any] = {}

    def _build_pipeline(**kwargs: Any) -> Pipeline:
        captured["build_kwargs"] = kwargs
        return base_pipeline

    def _add_model_to_pipeline(pipeline: Pipeline, model_arg: Any) -> Pipeline:
        captured["add_args"] = {"pipeline": pipeline, "model": model_arg}
        return final_pipeline

    monkeypatch.setattr(build_module, "build_pipeline", _build_pipeline)
    monkeypatch.setattr(build_module, "add_model_to_pipeline", _add_model_to_pipeline)

    result = build_module.build_pipeline_with_model(
        model_cfg=model_cfg,
        pipeline_cfg=pipeline_cfg,
        input_schema=input_schema,
        derived_schema=derived_schema,
        model=model,
    )

    assert result is final_pipeline
    assert captured["build_kwargs"]["model_cfg"] is model_cfg
    assert captured["build_kwargs"]["pipeline_cfg"] == pipeline_cfg
    assert captured["build_kwargs"]["input_schema"].equals(input_schema)
    assert captured["build_kwargs"]["derived_schema"].equals(derived_schema)
    assert captured["add_args"]["pipeline"] is base_pipeline
    assert captured["add_args"]["model"] is model


def test_build_pipeline_with_model_accepts_catboost_regressor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept CatBoost regressor instances and delegate model insertion to composition helper."""
    model = CatBoostRegressor(verbose=False)

    build_module = _import_build_module()

    monkeypatch.setattr(build_module, "build_pipeline", lambda **_: Pipeline(steps=[]))

    calls: list[Any] = []

    def _add_model_to_pipeline(pipeline: Pipeline, model_arg: Any) -> Pipeline:
        calls.append(model_arg)
        return Pipeline(steps=[("Model", model_arg)])

    monkeypatch.setattr(build_module, "add_model_to_pipeline", _add_model_to_pipeline)

    result = build_module.build_pipeline_with_model(
        model_cfg=_dummy_model_cfg(),
        pipeline_cfg={},
        input_schema=pd.DataFrame(),
        derived_schema=pd.DataFrame(),
        model=model,
    )

    assert isinstance(result, Pipeline)
    assert calls == [model]


def test_build_pipeline_with_model_rejects_non_catboost_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ``PipelineContractError`` when model is not CatBoost classifier/regressor."""
    build_module = _import_build_module()

    monkeypatch.setattr(build_module, "build_pipeline", lambda **_: Pipeline(steps=[]))

    add_model_calls: list[str] = []
    monkeypatch.setattr(
        build_module,
        "add_model_to_pipeline",
        lambda *_: add_model_calls.append("called") or Pipeline(steps=[]),
    )

    with pytest.raises(PipelineContractError, match="not a CatBoostClassifier or CatBoostRegressor"):
        build_module.build_pipeline_with_model(
            model_cfg=_dummy_model_cfg(),
            pipeline_cfg={},
            input_schema=pd.DataFrame(),
            derived_schema=pd.DataFrame(),
            model=cast(Any, object()),
        )

    assert add_model_calls == []
