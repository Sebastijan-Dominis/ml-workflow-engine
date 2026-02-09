"""Unit tests for Pydantic configuration schemas used in training.

These tests ensure minimal configs validate correctly and that
missing required fields raise appropriate errors.
"""

import pytest

from ml.runners.training.schemas.train_config_schema import (
    ConfigSchema,
)


def test_config_schema_valid(minimal_training_cfg: dict) -> None:
    """Validate that a minimal configuration passes schema validation."""

    validated = ConfigSchema(**minimal_training_cfg)
    assert validated.name == minimal_training_cfg["name"]
    assert validated.data.features_version == "test"


def test_config_schema_missing_required_field_causes_error(minimal_training_cfg: dict) -> None:
    """Ensure removing a required field triggers validation failure."""

    del minimal_training_cfg["data"]["train_file"]
    with pytest.raises(Exception):
        ConfigSchema(**minimal_training_cfg)