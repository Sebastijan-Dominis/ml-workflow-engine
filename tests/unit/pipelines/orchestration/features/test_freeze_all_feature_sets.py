"""Unit tests for feature-freezing batch orchestration."""

from __future__ import annotations

import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipelines.orchestration.features import freeze_all_feature_sets

pytestmark = pytest.mark.unit


def test_parse_args_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse CLI defaults for logging level, owner, and skip behavior."""
    monkeypatch.setattr(freeze_all_feature_sets.sys, "argv", ["freeze_all_feature_sets"])

    args = freeze_all_feature_sets.parse_args()

    assert args.logging_level == "INFO"
    assert args.owner == "Sebastijan"
    assert args.skip_if_existing is True


def test_parse_args_converts_skip_if_existing_from_cli_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert textual bool input for `--skip-if-existing` via shared parser."""
    monkeypatch.setattr(
        freeze_all_feature_sets.sys,
        "argv",
        [
            "freeze_all_feature_sets",
            "--logging-level",
            "DEBUG",
            "--owner",
            "CI",
            "--skip-if-existing",
            "false",
        ],
    )

    args = freeze_all_feature_sets.parse_args()

    assert args.logging_level == "DEBUG"
    assert args.owner == "CI"
    assert args.skip_if_existing is False


def test_main_returns_one_when_feature_registry_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail fast with exit code `1` when feature registry cannot be loaded."""
    monkeypatch.setattr(
        freeze_all_feature_sets,
        "parse_args",
        lambda: Namespace(logging_level="INFO", owner="Sebastijan", skip_if_existing=True),
    )
    monkeypatch.setattr(freeze_all_feature_sets, "iso_no_colon", lambda _dt: "20260306T130000")
    monkeypatch.setattr(freeze_all_feature_sets, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))
    monkeypatch.setattr(freeze_all_feature_sets, "setup_logging", lambda **kwargs: None)

    def _raise(_path: Path) -> dict[str, Any]:
        raise OSError("cannot read file")

    monkeypatch.setattr(freeze_all_feature_sets, "load_yaml", _raise)

    code = freeze_all_feature_sets.main()

    assert code == 1


def test_main_skips_existing_freezes_when_skip_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Skip subprocess execution when any freeze folder already exists for a feature set."""
    monkeypatch.chdir(tmp_path)
    freeze_dir = tmp_path / "feature_store" / "booking_context_features" / "v1"
    (freeze_dir / "existing_run").mkdir(parents=True)

    monkeypatch.setattr(
        freeze_all_feature_sets,
        "parse_args",
        lambda: Namespace(logging_level="INFO", owner="Sebastijan", skip_if_existing=True),
    )
    monkeypatch.setattr(freeze_all_feature_sets, "iso_no_colon", lambda _dt: "20260306T130500")
    monkeypatch.setattr(freeze_all_feature_sets, "uuid4", lambda: SimpleNamespace(hex="1122334455667788"))
    monkeypatch.setattr(freeze_all_feature_sets, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        freeze_all_feature_sets,
        "load_yaml",
        lambda _path: {"booking_context_features": ["v1"]},
    )

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(freeze_all_feature_sets.subprocess, "run", _fake_run)

    code = freeze_all_feature_sets.main()

    assert code == 0
    assert calls == []


def test_main_runs_freeze_command_for_each_feature_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Execute one freeze subprocess per discovered feature/version pair."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        freeze_all_feature_sets,
        "parse_args",
        lambda: Namespace(logging_level="debug", owner="CI", skip_if_existing=False),
    )
    monkeypatch.setattr(freeze_all_feature_sets, "iso_no_colon", lambda _dt: "20260306T131000")
    monkeypatch.setattr(freeze_all_feature_sets, "uuid4", lambda: SimpleNamespace(hex="9988776655443322"))
    monkeypatch.setattr(freeze_all_feature_sets, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        freeze_all_feature_sets,
        "load_yaml",
        lambda _path: {
            "booking_context_features": ["v1", "v2"],
            "pricing_party_features": ["vA"],
        },
    )

    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(freeze_all_feature_sets.subprocess, "run", _fake_run)

    code = freeze_all_feature_sets.main()

    assert code == 0
    assert len(commands) == 3
    assert commands[0][:3] == [freeze_all_feature_sets.sys.executable, "-m", "pipelines.features.freeze"]
    assert "--logging-level" in commands[0]
    assert "DEBUG" in commands[0]
    assert commands[0][-2:] == ["--owner", "CI"]


def test_main_falls_back_to_info_when_logging_level_is_unrecognized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use INFO log level fallback when args contain a non-standard logging level value."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        freeze_all_feature_sets,
        "parse_args",
        lambda: Namespace(logging_level="not-a-level", owner="Sebastijan", skip_if_existing=True),
    )
    monkeypatch.setattr(freeze_all_feature_sets, "iso_no_colon", lambda _dt: "20260306T131250")
    monkeypatch.setattr(freeze_all_feature_sets, "uuid4", lambda: SimpleNamespace(hex="0011001100110011"))
    monkeypatch.setattr(
        freeze_all_feature_sets,
        "load_yaml",
        lambda _path: {},
    )

    captured: dict[str, Any] = {}

    def _setup_logging(*, path: Path, level: int) -> None:
        captured["path"] = path
        captured["level"] = level

    monkeypatch.setattr(freeze_all_feature_sets, "setup_logging", _setup_logging)

    code = freeze_all_feature_sets.main()

    assert code == 0
    assert captured["path"].as_posix().endswith("freeze_all.log")
    assert captured["level"] == freeze_all_feature_sets.logging.INFO


def test_main_returns_subprocess_error_code_on_first_failed_freeze(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stop at first freeze failure and return delegated subprocess error code."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        freeze_all_feature_sets,
        "parse_args",
        lambda: Namespace(logging_level="INFO", owner="Sebastijan", skip_if_existing=False),
    )
    monkeypatch.setattr(freeze_all_feature_sets, "iso_no_colon", lambda _dt: "20260306T131500")
    monkeypatch.setattr(freeze_all_feature_sets, "uuid4", lambda: SimpleNamespace(hex="abcdabcdabcdabcd"))
    monkeypatch.setattr(freeze_all_feature_sets, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        freeze_all_feature_sets,
        "load_yaml",
        lambda _path: {"booking_context_features": ["v1", "v2"]},
    )

    attempts = {"count": 0}
    completion_messages: list[str] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        attempts["count"] += 1
        if attempts["count"] == 2:
            raise subprocess.CalledProcessError(
                returncode=17,
                cmd=cmd,
                stderr="first line\nfull traceback...",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(freeze_all_feature_sets.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        freeze_all_feature_sets,
        "log_completion",
        lambda _start, message: completion_messages.append(message),
    )

    code = freeze_all_feature_sets.main()

    assert code == 17
    assert attempts["count"] == 2
    assert completion_messages == ["Script terminated after successfully freezing 1 feature sets"]


def test_main_handles_called_process_error_without_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Return failing code even when subprocess error has empty stderr payload."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        freeze_all_feature_sets,
        "parse_args",
        lambda: Namespace(logging_level="INFO", owner="Sebastijan", skip_if_existing=False),
    )
    monkeypatch.setattr(freeze_all_feature_sets, "iso_no_colon", lambda _dt: "20260306T131550")
    monkeypatch.setattr(freeze_all_feature_sets, "uuid4", lambda: SimpleNamespace(hex="ffeeffeeffeeffee"))
    monkeypatch.setattr(freeze_all_feature_sets, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        freeze_all_feature_sets,
        "load_yaml",
        lambda _path: {"booking_context_features": ["v1"]},
    )

    def _raise_called_process_error(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        _ = kwargs
        raise subprocess.CalledProcessError(returncode=23, cmd=cmd, stderr="")

    monkeypatch.setattr(freeze_all_feature_sets.subprocess, "run", _raise_called_process_error)

    with caplog.at_level("ERROR", logger=freeze_all_feature_sets.__name__):
        code = freeze_all_feature_sets.main()

    assert code == 23
    assert "Failed to freeze 'booking_context_features' version 'v1': " in caplog.text
