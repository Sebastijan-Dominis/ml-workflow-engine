"""Tests for `DatasetConfig` field validators (merge_key, merge_how, merge_validate)."""

from __future__ import annotations

import pytest
from ml.exceptions import ConfigError
from ml.feature_freezing.freeze_strategies.tabular.config.models import DatasetConfig

pytestmark = pytest.mark.unit


def test_dataset_config_normalizes_merge_key_to_list() -> None:
    """When provided as a string, `merge_key` is converted to a list internally."""
    cfg = DatasetConfig.model_validate({
        "name": "ds",
        "version": "v1",
        "format": "csv",
        "merge_key": "row_id",
        "path_suffix": "data.{format}",
    })

    assert isinstance(cfg.merge_key, list)
    assert cfg.merge_key == ["row_id"]


def test_dataset_config_rejects_invalid_merge_how() -> None:
    """Values outside the allowed merge types should raise ConfigError."""
    payload = {
        "name": "ds",
        "version": "v1",
        "format": "csv",
        "merge_key": ["row_id"],
        "merge_how": "cross",
        "path_suffix": "data.{format}",
    }

    with pytest.raises(ConfigError, match="Invalid merge_how"):
        DatasetConfig.model_validate(payload)


def test_dataset_config_normalizes_and_rejects_invalid_merge_validate() -> None:
    """Known textual forms are normalized; unknown values raise ConfigError."""
    p1 = {
        "name": "ds",
        "version": "v1",
        "format": "csv",
        "merge_key": ["row_id"],
        "merge_validate": "one_to_one",
        "path_suffix": "data.{format}",
    }
    cfg = DatasetConfig.model_validate(p1)
    assert cfg.merge_validate == "1:1"

    p2 = p1.copy()
    p2["merge_validate"] = "not_a_valid_option"
    # pydantic may raise a ValidationError for literal mismatch before our
    # custom validator is invoked; accept either error class here.
    from pydantic import ValidationError

    with pytest.raises((ConfigError, ValidationError)):
        DatasetConfig.model_validate(p2)
