from __future__ import annotations

from datetime import datetime

import pytest
from ml.exceptions import ConfigError
from ml.pipelines.models import PipelineConfig

pytestmark = pytest.mark.integration


def _base_cfg_dict() -> dict:
    return {
        "name": "p",
        "version": "v1",
        "steps": ["SchemaValidator"],
        "assumptions": {
            "handles_categoricals": True,
            "supports_regression": True,
            "supports_classification": True,
        },
        "lineage": {"created_by": "t", "created_at": datetime.utcnow()},
    }


def test_pipeline_config_validates_successfully() -> None:
    PipelineConfig.model_validate(_base_cfg_dict())


def test_pipeline_config_rejects_invalid_version() -> None:
    bad = _base_cfg_dict()
    bad["version"] = "1"
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(bad)


def test_pipeline_config_rejects_unknown_steps() -> None:
    bad = _base_cfg_dict()
    bad["steps"] = ["UnknownStep"]
    with pytest.raises(ConfigError):
        PipelineConfig.model_validate(bad)
