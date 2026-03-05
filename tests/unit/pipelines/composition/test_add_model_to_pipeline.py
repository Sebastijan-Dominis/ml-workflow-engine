"""Unit tests for appending validated estimators to sklearn pipelines."""

import importlib
import sys
import types

import pytest
from ml.exceptions import PipelineContractError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

pytestmark = pytest.mark.unit


def _import_add_model_to_pipeline_with_registry(registry: dict[str, object]):
    """Import target module with an isolated model registry to avoid side effects."""
    module_name = "ml.pipelines.composition.add_model_to_pipeline"
    registries_name = "ml.registries"
    catalogs_name = "ml.registries.catalogs"

    sys.modules.pop(module_name, None)

    fake_registries = types.ModuleType(registries_name)
    fake_registries.__path__ = []  # Mark as package for submodule imports.
    sys.modules[registries_name] = fake_registries

    fake_catalogs = types.ModuleType(catalogs_name)
    fake_catalogs.__dict__["MODEL_CLASS_REGISTRY"] = registry
    sys.modules[catalogs_name] = fake_catalogs

    return importlib.import_module(module_name)


class _SupportedModel:
    """Registered model stub for positive-path validation."""


class _UnsupportedModel:
    """Simple non-registered model stub for negative-path validation."""


def test_add_model_to_pipeline_appends_model_as_final_model_step() -> None:
    """Return a new pipeline with a trailing `Model` step while preserving existing steps."""
    add_model_module = _import_add_model_to_pipeline_with_registry(
        {"Supported": _SupportedModel}
    )

    base_pipeline = Pipeline(
        steps=[("identity", FunctionTransformer(validate=False))]
    )
    model = _SupportedModel()

    result = add_model_module.add_model_to_pipeline(base_pipeline, model)

    assert result is not base_pipeline
    assert [name for name, _ in base_pipeline.steps] == ["identity"]
    assert [name for name, _ in result.steps] == ["identity", "Model"]
    assert result.named_steps["Model"] is model


def test_add_model_to_pipeline_rejects_unregistered_model_type() -> None:
    """Raise `PipelineContractError` when model type is not part of registry catalog."""
    add_model_module = _import_add_model_to_pipeline_with_registry(
        {"Supported": _SupportedModel}
    )

    base_pipeline = Pipeline(
        steps=[("identity", FunctionTransformer(validate=False))]
    )

    with pytest.raises(PipelineContractError, match="not supported"):
        add_model_module.add_model_to_pipeline(base_pipeline, _UnsupportedModel())
