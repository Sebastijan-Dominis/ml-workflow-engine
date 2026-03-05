"""Unit tests for model-to-pipeline compatibility contract validation."""

from dataclasses import dataclass
from typing import cast

import pytest
from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import PipelineContractError
from ml.features.validation.validate_contract import validate_model_feature_pipeline_contract

pytestmark = pytest.mark.unit

ModelConfig = SearchModelConfig | TrainModelConfig


@dataclass
class _TaskStub:
    """Minimal task descriptor used by model config stubs."""

    type: str


@dataclass
class _ModelConfigStub:
    """Minimal model config stub exposing only attributes read by the validator."""

    task: _TaskStub
    algorithm: str


def _as_model_config(config: _ModelConfigStub) -> ModelConfig:
    """Cast lightweight stub config to validator function's expected model union."""
    return cast(ModelConfig, config)


def test_validate_contract_accepts_supported_classification_pipeline() -> None:
    """Pass when the pipeline explicitly supports the model task type."""
    model_cfg = _ModelConfigStub(task=_TaskStub(type="classification"), algorithm="xgboost")
    pipeline_cfg = {"assumptions": {"supports_classification": True, "supports_regression": False}}

    validate_model_feature_pipeline_contract(_as_model_config(model_cfg), pipeline_cfg)


def test_validate_contract_raises_when_pipeline_does_not_support_task_type() -> None:
    """Reject model/pipeline pairings where task support is absent."""
    model_cfg = _ModelConfigStub(task=_TaskStub(type="regression"), algorithm="xgboost")
    pipeline_cfg = {"assumptions": {"supports_classification": True, "supports_regression": False}}

    with pytest.raises(PipelineContractError, match="Pipeline does not support the task type"):
        validate_model_feature_pipeline_contract(_as_model_config(model_cfg), pipeline_cfg)


def test_validate_contract_requires_cat_features_for_catboost() -> None:
    """Require explicit categorical feature list when algorithm is CatBoost."""
    model_cfg = _ModelConfigStub(task=_TaskStub(type="classification"), algorithm="catboost")
    pipeline_cfg = {
        "assumptions": {
            "supports_classification": True,
            "supports_regression": False,
            "handles_categoricals": True,
        }
    }

    with pytest.raises(PipelineContractError, match="Categorical features must be provided"):
        validate_model_feature_pipeline_contract(_as_model_config(model_cfg), pipeline_cfg, cat_features=None)


def test_validate_contract_requires_pipeline_categorical_support_for_catboost() -> None:
    """Reject CatBoost when pipeline assumptions do not advertise categorical handling."""
    model_cfg = _ModelConfigStub(task=_TaskStub(type="classification"), algorithm="catboost")
    pipeline_cfg = {
        "assumptions": {
            "supports_classification": True,
            "supports_regression": False,
            "handles_categoricals": False,
        }
    }

    with pytest.raises(PipelineContractError, match="does not support categorical features"):
        validate_model_feature_pipeline_contract(
            _as_model_config(model_cfg),
            pipeline_cfg,
            cat_features=["market_segment"],
        )


def test_validate_contract_accepts_catboost_when_all_constraints_are_met() -> None:
    """Pass CatBoost validation when task support and categorical requirements are satisfied."""
    model_cfg = _ModelConfigStub(task=_TaskStub(type="classification"), algorithm="catboost")
    pipeline_cfg = {
        "assumptions": {
            "supports_classification": True,
            "supports_regression": False,
            "handles_categoricals": True,
        }
    }

    validate_model_feature_pipeline_contract(
        _as_model_config(model_cfg),
        pipeline_cfg,
        cat_features=["market_segment", "distribution_channel"],
    )
