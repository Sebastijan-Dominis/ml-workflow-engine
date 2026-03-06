"""Unit tests for top-level data config validation dispatch."""

from __future__ import annotations

import ml.data.config.validate_config as validate_config_module
import pytest
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


class _CapturingSchema:
    """Schema stub that captures constructor kwargs for assertions."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        self.__class__.last_kwargs = kwargs


def test_validate_config_dispatches_to_interim_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiate ``InterimConfig`` when type selector is ``interim``."""
    monkeypatch.setattr(validate_config_module, "InterimConfig", _CapturingSchema)

    payload = {"raw_data_version": "v1", "cleaning": {"lowercase_columns": True}}

    result = validate_config_module.validate_config(payload, "interim")

    assert isinstance(result, _CapturingSchema)
    assert _CapturingSchema.last_kwargs == payload


def test_validate_config_dispatches_to_processed_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiate ``ProcessedConfig`` when type selector is ``processed``."""
    monkeypatch.setattr(validate_config_module, "ProcessedConfig", _CapturingSchema)

    payload = {"data": {"name": "hotel_bookings"}, "lineage": {"created_by": "tests"}}

    result = validate_config_module.validate_config(payload, "processed")

    assert isinstance(result, _CapturingSchema)
    assert _CapturingSchema.last_kwargs == payload


def test_validate_config_wraps_schema_validation_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap schema constructor failures in a stable project-level ``ConfigError``."""

    class _FailingSchema:
        def __init__(self, **_kwargs) -> None:
            raise ValueError("bad payload")

    monkeypatch.setattr(validate_config_module, "InterimConfig", _FailingSchema)

    with pytest.raises(ConfigError, match="Configuration validation error") as exc_info:
        validate_config_module.validate_config({}, "interim")

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_validate_config_wraps_unsupported_type_with_cause() -> None:
    """Raise wrapper ``ConfigError`` for unsupported type selectors and preserve cause."""
    with pytest.raises(ConfigError, match="Configuration validation error") as exc_info:
        validate_config_module.validate_config({}, "unknown")  # type: ignore[arg-type]

    assert isinstance(exc_info.value.__cause__, ConfigError)
    assert "Unsupported config type" in str(exc_info.value.__cause__)
