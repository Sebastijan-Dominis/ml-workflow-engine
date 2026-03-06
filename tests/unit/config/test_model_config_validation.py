"""Unit tests for model-configuration validation entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from ml.config import validation as validation_module
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


@dataclass
class _DummyMeta:
    """Lightweight metadata holder matching the fields touched by validator."""

    validation_status: str | None = None
    validation_errors: Any = "unchanged"


@dataclass
class _DummyConfig:
    """Config stub that mimics pydantic model object shape used by helper."""

    meta: _DummyMeta


class _FakeValidationError(Exception):
    """ValidationError test double exposing ``errors()`` payload."""

    def __init__(self, payload: list[dict[str, Any]]) -> None:
        super().__init__("validation failed")
        self._payload = payload

    def errors(self) -> list[dict[str, Any]]:
        """Return pydantic-like error entries consumed by logger logic."""
        return self._payload


def test_validate_model_config_search_marks_meta_ok_and_clears_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate search config and normalize metadata fields on success."""
    cfg_raw = {"a": 1}
    dummy = _DummyConfig(meta=_DummyMeta(validation_status="old", validation_errors=[{"x": 1}]))

    def _build_search_cfg(**kwargs: Any) -> _DummyConfig:
        assert kwargs == {"a": 1, "_meta": {}}
        return dummy

    monkeypatch.setattr(validation_module, "SearchModelConfig", _build_search_cfg)

    result = validation_module.validate_model_config(cfg_raw, "search")

    assert result is dummy
    assert result.meta.validation_status == "ok"
    assert result.meta.validation_errors is None
    assert cfg_raw["_meta"] == {}


def test_validate_model_config_train_marks_meta_ok_and_preserves_existing_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate train config while retaining caller-provided ``_meta`` mapping object."""
    existing_meta: dict[str, Any] = {"note": "keep"}
    cfg_raw: dict[str, Any] = {"_meta": existing_meta, "train": True}
    dummy = _DummyConfig(meta=_DummyMeta())

    def _build_train_cfg(**kwargs: Any) -> _DummyConfig:
        assert kwargs == cfg_raw
        assert kwargs["_meta"] is existing_meta
        return dummy

    monkeypatch.setattr(validation_module, "TrainModelConfig", _build_train_cfg)

    result = validation_module.validate_model_config(cfg_raw, "train")

    assert result is dummy
    assert result.meta.validation_status == "ok"
    assert result.meta.validation_errors is None
    assert cfg_raw["_meta"] is existing_meta
    assert cfg_raw["_meta"] == {"note": "keep"}


def test_validate_model_config_rejects_unknown_type_and_marks_raw_meta_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reject unknown config type with explicit ConfigError and error log."""
    cfg_raw: dict[str, Any] = {}

    with caplog.at_level("ERROR", logger=validation_module.__name__), pytest.raises(
        ConfigError,
        match="Unknown config type: unknown",
    ):
        validation_module.validate_model_config(cfg_raw, "unknown")  # type: ignore[arg-type]

    assert cfg_raw["_meta"]["validation_status"] == "failed"
    assert "Unknown config type: unknown" in caplog.text


def test_validate_model_config_wraps_validation_error_and_stores_error_payload(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wrap schema validation failures and store field-level details in ``_meta``."""
    cfg_raw: dict[str, Any] = {"bad": "payload"}
    errors_payload = [
        {"loc": ("task", "type"), "msg": "Field required"},
        {"loc": ("split",), "msg": "Invalid value"},
    ]

    monkeypatch.setattr(validation_module, "ValidationError", _FakeValidationError)
    monkeypatch.setattr(
        validation_module,
        "SearchModelConfig",
        lambda **kwargs: (_ for _ in ()).throw(_FakeValidationError(errors_payload)),
    )

    with caplog.at_level("ERROR", logger=validation_module.__name__), pytest.raises(
        ConfigError,
        match="Validation failed for search config",
    ) as exc_info:
        validation_module.validate_model_config(cfg_raw, "search")

    assert isinstance(exc_info.value.__cause__, _FakeValidationError)
    assert cfg_raw["_meta"]["validation_status"] == "failed"
    assert cfg_raw["_meta"]["validation_errors"] == errors_payload
    assert "Model config validation failed for type 'search':" in caplog.text
    assert " - Field 'task.type': Field required" in caplog.text
    assert " - Field 'split': Invalid value" in caplog.text
