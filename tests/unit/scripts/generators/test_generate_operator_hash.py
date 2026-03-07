"""Unit tests for the operator-hash generator CLI script."""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch):
    """Import the script module with a lightweight registry stub for isolation."""
    fake_catalogs = types.ModuleType("ml.registries.catalogs")
    fake_catalogs.__dict__["FEATURE_OPERATORS"] = {
        "ArrivalDate": object(),
        "TotalStay": object(),
    }
    monkeypatch.setitem(sys.modules, "ml.registries.catalogs", fake_catalogs)

    sys.modules.pop("scripts.generators.generate_operator_hash", None)
    imported = importlib.import_module("scripts.generators.generate_operator_hash")
    yield imported
    sys.modules.pop("scripts.generators.generate_operator_hash", None)


def test_parse_args_reads_operator_list(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse ``--operators`` values into the expected CLI namespace."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_operator_hash",
            "--operators",
            "ArrivalDate",
            "TotalStay",
        ],
    )

    args = module.parse_args()

    assert args.operators == ["ArrivalDate", "TotalStay"]


def test_main_returns_success_and_logs_hash(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return ``0`` and print hash output when generation succeeds."""
    monkeypatch.setattr(module, "parse_args", lambda: argparse.Namespace(operators=["ArrivalDate"]))
    monkeypatch.setattr(module, "generate_operator_hash", lambda names: "hash123")

    class _FakeUuid:
        hex = "1234567890abcdef"

    monkeypatch.setattr(module, "uuid4", lambda: _FakeUuid())
    monkeypatch.setattr(module, "iso_no_colon", lambda dt: "20260307T010203")

    setup_calls: dict[str, object] = {}

    def _fake_setup_logging(path: Path, level: int) -> None:
        setup_calls["path"] = path
        setup_calls["level"] = level

    monkeypatch.setattr(module, "setup_logging", _fake_setup_logging)

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Generated operators hash for operator names ['ArrivalDate']: hash123" in captured.out
    path_value = setup_calls["path"]
    assert isinstance(path_value, Path)
    assert (
        "scripts_logs/generators/generate_operator_hash/"
        in path_value.as_posix()
    )


def test_main_returns_failure_when_generation_raises(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return ``1`` when operator hash generation raises an exception."""
    monkeypatch.setattr(module, "parse_args", lambda: argparse.Namespace(operators=["ArrivalDate"]))

    def _boom(names: list[str]) -> str:
        raise RuntimeError("generation failed")

    monkeypatch.setattr(module, "generate_operator_hash", _boom)
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    class _FakeUuid:
        hex = "1234567890abcdef"

    monkeypatch.setattr(module, "uuid4", lambda: _FakeUuid())
    monkeypatch.setattr(module, "iso_no_colon", lambda dt: "20260307T010203")

    assert module.main() == 1
