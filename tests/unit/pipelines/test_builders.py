"""Unit tests for pipeline builder orchestration behavior."""

from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from ml.exceptions import ConfigError
from ml.pipelines.models import PipelineConfig

pytestmark = pytest.mark.unit


class _SchemaValidator:
    """Stub component capturing required feature inputs."""

    def __init__(self, *, required_features: list[str]) -> None:
        self.required_features = required_features


class _FillCategoricalMissing:
    """Stub component capturing categorical feature inputs."""

    def __init__(self, *, categorical_features: list[str]) -> None:
        self.categorical_features = categorical_features


class _FeatureEngineer:
    """Stub component capturing derived schema and operator mapping."""

    def __init__(self, *, derived_schema: pd.DataFrame, operators: dict[str, object]) -> None:
        self.derived_schema = derived_schema
        self.operators = operators


class _FeatureSelector:
    """Stub component capturing selected feature inputs."""

    def __init__(self, *, selected_features: list[str]) -> None:
        self.selected_features = selected_features




def _as_pipeline_config(cfg_dict: dict) -> PipelineConfig:
    """Convert lightweight test dict to a PipelineConfig Pydantic model with minimal defaults."""
    # Fill required fields if missing
    if "name" not in cfg_dict:
        cfg_dict["name"] = "test_pipeline"
    if "version" not in cfg_dict:
        cfg_dict["version"] = "v1"
    if "steps" not in cfg_dict:
        cfg_dict["steps"] = []
    if "assumptions" not in cfg_dict:
        cfg_dict["assumptions"] = {
            "handles_categoricals": True,
            "supports_regression": True,
            "supports_classification": True,
        }
    else:
        for k in ["handles_categoricals", "supports_regression", "supports_classification"]:
            cfg_dict["assumptions"].setdefault(k, False)
    if "lineage" not in cfg_dict:
        cfg_dict["lineage"] = {
            "created_by": "test_user",
            "created_at": datetime.utcnow()
        }

    return PipelineConfig.model_validate(cfg_dict)

def _import_builders_with_stubs(
    *,
    features: object,
    operators: dict[str, object],
    pipeline_components: dict[str, object],
) -> tuple[types.ModuleType, list[pd.DataFrame], list[tuple[object, pd.DataFrame, pd.DataFrame]]]:
    """Import builders module with controlled stub dependencies for deterministic testing."""
    module_name = "ml.pipelines.builders"
    registries_name = "ml.registries"
    catalogs_name = "ml.registries.catalogs"
    schema_utils_name = "ml.pipelines.schema_utils"
    operator_factory_name = "ml.pipelines.operator_factory"

    sys.modules.pop(module_name, None)

    schema_calls: list[tuple[object, pd.DataFrame, pd.DataFrame]] = []
    operator_calls: list[pd.DataFrame] = []

    fake_schema_utils = types.ModuleType(schema_utils_name)

    def _fake_get_pipeline_features(*, model_cfg: object, input_schema: pd.DataFrame, derived_schema: pd.DataFrame):
        schema_calls.append((model_cfg, input_schema, derived_schema))
        return features

    fake_schema_utils.__dict__["get_pipeline_features"] = _fake_get_pipeline_features
    sys.modules[schema_utils_name] = fake_schema_utils

    fake_operator_factory = types.ModuleType(operator_factory_name)

    def _fake_build_operators(derived_schema: pd.DataFrame):
        operator_calls.append(derived_schema)
        return operators

    fake_operator_factory.__dict__["build_operators"] = _fake_build_operators
    sys.modules[operator_factory_name] = fake_operator_factory

    fake_registries = types.ModuleType(registries_name)
    fake_registries.__path__ = []
    sys.modules[registries_name] = fake_registries

    fake_catalogs = types.ModuleType(catalogs_name)
    fake_catalogs.__dict__["PIPELINE_COMPONENTS"] = pipeline_components
    sys.modules[catalogs_name] = fake_catalogs

    module = importlib.import_module(module_name)
    return module, operator_calls, schema_calls


def test_build_pipeline_wires_components_with_expected_inputs_and_order() -> None:
    """Construct configured steps in order, lower-case names, and expected constructor payloads."""
    input_schema = pd.DataFrame({"feature": ["f1"], "dtype": ["float64"]})
    derived_schema = pd.DataFrame({"feature": ["f2"], "source_operator": ["op_a"]})
    model_cfg = object()
    features = SimpleNamespace(
        input_features=["f1"],
        selected_features=["f1", "f2"],
        categorical_features=["cat_a"],
    )
    operators = {"op_a": object()}

    builders, operator_calls, schema_calls = _import_builders_with_stubs(
        features=features,
        operators=operators,
        pipeline_components={
            "SchemaValidator": _SchemaValidator,
            "FillCategoricalMissing": _FillCategoricalMissing,
            "FeatureEngineer": _FeatureEngineer,
            "FeatureSelector": _FeatureSelector,
            "Model": None,
        },
    )

    pipeline = builders.build_pipeline(
        model_cfg=model_cfg,
        pipeline_cfg=_as_pipeline_config({
            "steps": [
                "SchemaValidator",
                "FillCategoricalMissing",
                "FeatureEngineer",
                "FeatureSelector",
                "Model",
            ]
        }),
        input_schema=input_schema,
        derived_schema=derived_schema,
    )

    assert [name for name, _ in pipeline.steps] == [
        "schemavalidator",
        "fillcategoricalmissing",
        "featureengineer",
        "featureselector",
    ]
    assert isinstance(pipeline.named_steps["schemavalidator"], _SchemaValidator)
    assert pipeline.named_steps["schemavalidator"].required_features == ["f1"]
    assert isinstance(pipeline.named_steps["fillcategoricalmissing"], _FillCategoricalMissing)
    assert pipeline.named_steps["fillcategoricalmissing"].categorical_features == ["cat_a"]
    assert isinstance(pipeline.named_steps["featureengineer"], _FeatureEngineer)
    assert pipeline.named_steps["featureengineer"].derived_schema.equals(derived_schema)
    assert pipeline.named_steps["featureengineer"].operators is operators
    assert isinstance(pipeline.named_steps["featureselector"], _FeatureSelector)
    assert pipeline.named_steps["featureselector"].selected_features == ["f1", "f2"]

    assert operator_calls == [derived_schema]
    assert schema_calls == [(model_cfg, input_schema, derived_schema)]


def test_build_pipeline_raises_config_error_for_unknown_step() -> None:
    """Fail fast with `ConfigError` when configuration includes an unregistered step name."""
    builders, _, _ = _import_builders_with_stubs(
        features=SimpleNamespace(
            input_features=[],
            selected_features=[],
            categorical_features=[],
        ),
        operators={},
        pipeline_components={
            "SchemaValidator": _SchemaValidator,
            "FillCategoricalMissing": _FillCategoricalMissing,
            "FeatureEngineer": _FeatureEngineer,
            "FeatureSelector": _FeatureSelector,
            "Model": None,
        },
    )

    with pytest.raises(ConfigError, match="Unknown pipeline steps: {'UnknownStep'}"):
        builders.build_pipeline(
            model_cfg=object(),
            pipeline_cfg=_as_pipeline_config({"steps": ["UnknownStep"]}),
            input_schema=pd.DataFrame({"feature": [], "dtype": []}),
            derived_schema=pd.DataFrame({"feature": [], "source_operator": []}),
        )


def test_build_pipeline_returns_empty_pipeline_when_no_steps_are_configured() -> None:
    """Return an empty sklearn Pipeline when the step list is absent."""
    builders, _, _ = _import_builders_with_stubs(
        features=SimpleNamespace(
            input_features=["a"],
            selected_features=["a"],
            categorical_features=[],
        ),
        operators={"noop": object()},
        pipeline_components={
            "SchemaValidator": _SchemaValidator,
            "FillCategoricalMissing": _FillCategoricalMissing,
            "FeatureEngineer": _FeatureEngineer,
            "FeatureSelector": _FeatureSelector,
            "Model": None,
        },
    )

    with pytest.raises(ConfigError, match="Pipeline steps cannot be empty"):
        builders.build_pipeline(
            model_cfg=object(),
            pipeline_cfg=_as_pipeline_config({
                "assumptions": {
                    "handles_categoricals": False,
                    "supports_regression": False,
                    "supports_classification": False,
                },
                "steps": [],
            }),
            input_schema=pd.DataFrame({"feature": ["a"], "dtype": ["int64"]}),
            derived_schema=pd.DataFrame({"feature": [], "source_operator": []}),
        )
