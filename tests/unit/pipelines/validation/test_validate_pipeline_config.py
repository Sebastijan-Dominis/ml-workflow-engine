"""Tests for pipeline config validation."""

import logging
from datetime import datetime
from unittest.mock import patch

import pytest
from ml.exceptions import ConfigError
from ml.pipelines.validation import validate_pipeline_config, validate_pipeline_config_consistency


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

class DummyMetadata:
    def __init__(self, pipeline_hash):
        self.pipeline_hash = pipeline_hash


class DummySearchRecord:
    def __init__(self, pipeline_hash):
        self.metadata = DummyMetadata(pipeline_hash)


def test_validate_pipeline_config_consistency_hash_match(tmp_path, caplog):
    """Verify the function completes successfully and logs a debug message when the pipeline hash matches the value in search metadata."""
    caplog.set_level(logging.DEBUG)

    search_dir = tmp_path
    actual_hash = "abc123"

    with patch("ml.pipelines.validation.load_json") as mock_load, \
         patch("ml.pipelines.validation.validate_search_record") as mock_validate:

        mock_load.return_value = {"dummy": "data"}
        mock_validate.return_value = DummySearchRecord("abc123")

        validate_pipeline_config_consistency(actual_hash, search_dir)

        mock_load.assert_called_once_with(search_dir / "metadata.json")
        mock_validate.assert_called_once()

        assert "Pipeline config hash matches search metadata." in caplog.text


def test_validate_pipeline_config_consistency_hash_mismatch(tmp_path):
    """Verify the function raises ConfigError when the provided pipeline hash differs from the hash stored in search metadata."""
    search_dir = tmp_path
    actual_hash = "abc123"

    with patch("ml.pipelines.validation.load_json") as mock_load, \
         patch("ml.pipelines.validation.validate_search_record") as mock_validate:

        mock_load.return_value = {"dummy": "data"}
        mock_validate.return_value = DummySearchRecord("different_hash")

        with pytest.raises(ConfigError) as excinfo:
            validate_pipeline_config_consistency(actual_hash, search_dir)

        assert "Pipeline config hash mismatch" in str(excinfo.value)

        mock_load.assert_called_once_with(search_dir / "metadata.json")
        mock_validate.assert_called_once()


def test_validate_pipeline_config_consistency_logs_error_on_mismatch(tmp_path, caplog):
    """Verify the function logs an error message when a pipeline hash mismatch occurs."""
    search_dir = tmp_path

    with patch("ml.pipelines.validation.load_json") as mock_load, \
         patch("ml.pipelines.validation.validate_search_record") as mock_validate:

        mock_load.return_value = {}
        mock_validate.return_value = DummySearchRecord("expected_hash")

        with pytest.raises(ConfigError):
            validate_pipeline_config_consistency("actual_hash", search_dir)

        assert "Pipeline config hash mismatch" in caplog.text
