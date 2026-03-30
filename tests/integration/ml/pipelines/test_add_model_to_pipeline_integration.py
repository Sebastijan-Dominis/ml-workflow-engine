from __future__ import annotations

import importlib
import sys
import types
from typing import Any, cast

import pytest
from ml.exceptions import PipelineContractError
from sklearn.pipeline import Pipeline

pytestmark = pytest.mark.integration


class DummyModel:
    pass


def _import_add_model_module_with_registry(mapping: dict[str, type]):
    # Ensure fresh import and inject a lightweight fake registries.catalogs
    mod_name = "ml.pipelines.composition.add_model_to_pipeline"
    registries_name = "ml.registries"
    catalogs_name = "ml.registries.catalogs"

    sys.modules.pop(mod_name, None)

    fake_registries = types.ModuleType(registries_name)
    fake_registries.__path__ = []
    sys.modules[registries_name] = fake_registries

    fake_catalogs = types.ModuleType(catalogs_name)
    cast(Any, fake_catalogs).MODEL_CLASS_REGISTRY = mapping
    sys.modules[catalogs_name] = fake_catalogs

    return importlib.import_module(mod_name)


def test_add_model_to_pipeline_appends_supported_model() -> None:
    mod = _import_add_model_module_with_registry({"Dummy": DummyModel})

    pipeline = Pipeline([("noop", object())])
    model = DummyModel()

    pipeline_with_model = mod.add_model_to_pipeline(pipeline, model)

    assert pipeline_with_model.steps[-1][0] == "Model"
    assert pipeline_with_model.steps[-1][1] is model


def test_add_model_to_pipeline_rejects_unsupported() -> None:
    mod = _import_add_model_module_with_registry({})

    with pytest.raises(PipelineContractError):
        mod.add_model_to_pipeline(Pipeline([("noop", object())]), object())
