"""Tests for pipeline config validation."""

from datetime import datetime

import pytest
from ml.exceptions import ConfigError
from ml.pipelines.validation import validate_pipeline_config


def sample_pipeline_config() -> dict:
    """Return a valid sample pipeline configuration dictionary."""
    return {
        "name": "example_pipeline",
        "version": "v1",
        "description": "A test pipeline",
        "steps": ["SchemaValidator", "FeatureEngineer", "Model"],
        "assumptions": {
            "handles_categoricals": True,
            "supports_regression": True,
            "supports_classification": False
        },
        "lineage": {
            "created_by": "test_user",
            "created_at": datetime.now().isoformat()
        }
    }



def test_empty_steps_raises():
    """Test that empty steps list raises ConfigError."""
    cfg_raw = sample_pipeline_config()
    cfg_raw["steps"] = []
    with pytest.raises(ConfigError, match="Pipeline config validation failed."):
        validate_pipeline_config(cfg_raw)


def test_unknown_step_raises():
    """Test that unknown steps raise ConfigError."""
    cfg_raw = sample_pipeline_config()
    cfg_raw["steps"].append("UnknownStep")
    with pytest.raises(ConfigError, match="Pipeline config validation failed."):
        validate_pipeline_config(cfg_raw)


@pytest.mark.parametrize(
    "missing_key",
    ["handles_categoricals", "supports_regression", "supports_classification"]
)
def test_missing_assumption_key_raises(missing_key):
    """Test that missing assumption keys raise ConfigError."""
    cfg_raw = sample_pipeline_config()
    del cfg_raw["assumptions"][missing_key]
    with pytest.raises(ConfigError, match="Pipeline config validation failed."):
        validate_pipeline_config(cfg_raw)


def test_lineage_validation():
    """Test that missing lineage fields raise ConfigError."""
    cfg_raw = sample_pipeline_config()
    del cfg_raw["lineage"]["created_by"]
    with pytest.raises(ConfigError, match="Pipeline config validation failed."):
        validate_pipeline_config(cfg_raw)
