"""Unit tests for shared logging configuration helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from ml import logging_config as module

pytestmark = pytest.mark.unit


def test_setup_logging_creates_parent_and_calls_basic_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Create parent directory and configure root logging with expected arguments."""
    log_path = tmp_path / "logs" / "app.log"

    captured: dict[str, Any] = {}

    def _basic_config(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(module.logging, "basicConfig", _basic_config)

    module.setup_logging(log_path, level=logging.DEBUG)

    assert log_path.parent.exists()
    assert captured["level"] == logging.DEBUG
    assert captured["filename"] == str(log_path)
    assert captured["format"] == module.LOG_FORMAT


def test_add_file_handler_builds_handler_and_attaches_to_root_logger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Create and register file handler with formatter and level settings."""
    log_path = tmp_path / "logs" / "extra.log"

    attached: list[logging.FileHandler] = []
    root_logger = logging.getLogger()
    original_add_handler = root_logger.addHandler

    def _record_add_handler(handler: logging.FileHandler) -> None:
        attached.append(handler)
        original_add_handler(handler)

    monkeypatch.setattr(root_logger, "addHandler", _record_add_handler)

    handler = module.add_file_handler(log_path, level=logging.WARNING)

    assert log_path.parent.exists()
    assert attached == [handler]
    assert handler.level == logging.WARNING

    handler.close()


def test_bootstrap_logging_calls_basic_config_without_filename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configure root logging for console usage without setting file destination."""
    captured: dict[str, Any] = {}

    def _basic_config(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(module.logging, "basicConfig", _basic_config)

    module.bootstrap_logging(level=logging.ERROR)

    assert captured["level"] == logging.ERROR
    assert captured["format"] == module.LOG_FORMAT
    assert "filename" not in captured
