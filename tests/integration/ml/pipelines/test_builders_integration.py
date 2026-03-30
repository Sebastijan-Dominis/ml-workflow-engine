from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.config.schemas.model_cfg import SearchModelConfig
from ml.pipelines.models import PipelineConfig

pytestmark = pytest.mark.integration


class _SchemaValidator:
    def __init__(self, *, required_features: list[str]) -> None:
        self.required_features = required_features


class _FillCategoricalMissing:
    def __init__(self, *, categorical_features: list[str]) -> None:
        self.categorical_features = categorical_features


class _FeatureEngineer:
    def __init__(self, *, derived_schema: pd.DataFrame, operators: dict[str, object]) -> None:
        self.derived_schema = derived_schema
        self.operators = operators


class _FeatureSelector:
    def __init__(self, *, selected_features: list[str]) -> None:
        self.selected_features = selected_features


def _pipeline_cfg() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "name": "p",
            "version": "v1",
            "steps": [
                "SchemaValidator",
                "FillCategoricalMissing",
                "FeatureEngineer",
                "FeatureSelector",
                "Model",
            ],
            "assumptions": {
                "handles_categoricals": True,
                "supports_regression": True,
                "supports_classification": True,
            },
            "lineage": {"created_by": "t", "created_at": "2026-01-01T00:00:00Z"},
        }
    )


def test_build_pipeline_wires_components_correctly(monkeypatch) -> None:
    input_schema = pd.DataFrame({"feature": ["f1"], "dtype": ["float64"]})
    derived_schema = pd.DataFrame({"feature": ["f2"], "source_operator": ["op_a"]})

    features = SimpleNamespace(
        input_features=["f1"], selected_features=["f1", "f2"], categorical_features=["cat_a"]
    )
    operators: dict[str, object] = {"op_a": object()}

    # Inject lightweight fake dependency modules to avoid heavy imports / circulars
    module_name = "ml.pipelines.builders"
    schema_utils_name = "ml.pipelines.schema_utils"
    operator_factory_name = "ml.pipelines.operator_factory"
    registries_name = "ml.registries"
    catalogs_name = "ml.registries.catalogs"

    sys.modules.pop(module_name, None)

    fake_schema = types.ModuleType(schema_utils_name)
    cast(Any, fake_schema).get_pipeline_features = lambda *a, **k: features
    sys.modules[schema_utils_name] = fake_schema

    fake_operator = types.ModuleType(operator_factory_name)
    cast(Any, fake_operator).build_operators = lambda ds: operators
    sys.modules[operator_factory_name] = fake_operator

    fake_registries = types.ModuleType(registries_name)
    fake_registries.__path__ = []
    sys.modules[registries_name] = fake_registries

    fake_catalogs = types.ModuleType(catalogs_name)
    cast(Any, fake_catalogs).PIPELINE_COMPONENTS = {
        "SchemaValidator": _SchemaValidator,
        "FillCategoricalMissing": _FillCategoricalMissing,
        "FeatureEngineer": _FeatureEngineer,
        "FeatureSelector": _FeatureSelector,
        "Model": None,
    }
    sys.modules[catalogs_name] = fake_catalogs

    builders = importlib.import_module(module_name)

    model_cfg = SimpleNamespace(segmentation=SimpleNamespace(enabled=False, include_in_model=False, filters=[]))
    pipeline_cfg = _pipeline_cfg()

    pipeline = builders.build_pipeline(
        model_cfg=cast(SearchModelConfig, model_cfg),
        pipeline_cfg=pipeline_cfg,
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
