"""Unit tests for the row-id fingerprint generator CLI script."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def module() -> Any:
    """Import the target script module for isolated monkeypatching per test."""
    imported = importlib.import_module("scripts.generators.generate_cols_for_row_id_fingerprint")
    yield imported


def test_main_returns_success_and_prints_fingerprint(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return ``0`` and emit fingerprint details when computation succeeds."""

    class _FakeUuid:
        hex = "1234567890abcdef"

    monkeypatch.setattr(module, "uuid4", lambda: _FakeUuid())
    monkeypatch.setattr(module, "iso_no_colon", lambda dt: "20260307T020304")
    monkeypatch.setattr(module, "cols_for_row_id", ["hotel", "arrival_date"])
    monkeypatch.setattr(module, "compute_cols_for_row_id_fingerprint", lambda cols: "fp123")

    setup_calls: dict[str, object] = {}

    def _fake_setup_logging(path: Path, level: int) -> None:
        setup_calls["path"] = path
        setup_calls["level"] = level

    monkeypatch.setattr(module, "setup_logging", _fake_setup_logging)

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Fingerprint for cols_for_row_id: fp123" in captured.out

    path_value = setup_calls["path"]
    assert isinstance(path_value, Path)
    assert "scripts_logs/generators/generate_cols_for_row_id_fingerprint/" in path_value.as_posix()


def test_main_returns_failure_when_fingerprint_computation_raises(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return ``1`` when fingerprint computation raises an exception."""

    class _FakeUuid:
        hex = "1234567890abcdef"

    monkeypatch.setattr(module, "uuid4", lambda: _FakeUuid())
    monkeypatch.setattr(module, "iso_no_colon", lambda dt: "20260307T020304")
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    def _boom(cols: list[str]) -> str:
        raise RuntimeError("fingerprint failed")

    monkeypatch.setattr(module, "compute_cols_for_row_id_fingerprint", _boom)

    assert module.main() == 1
