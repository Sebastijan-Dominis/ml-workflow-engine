"""E2E smoke tests for experiment orchestration CLI boundary behavior."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
from pipelines.orchestration.experiments import execute_experiment_with_latest as module

pytestmark = pytest.mark.e2e


def test_execute_experiment_with_latest_main_returns_stage_failure_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return failing stage code and stop pipeline after first subprocess failure."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "execute_experiment_with_latest",
            "--problem",
            "cancellation",
            "--segment",
            "global",
            "--version",
            "v1",
            "--env",
            "test",
            "--strict",
            "true",
            "--logging-level",
            "INFO",
            "--owner",
            "CI",
            "--clean-up-failure-management",
            "true",
            "--overwrite-existing",
            "false",
        ],
    )

    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260307T120000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    completion_messages: list[str] = []
    run_calls: list[list[str]] = []

    monkeypatch.setattr(module, "log_completion", lambda _start, message: completion_messages.append(message))

    def _run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        run_calls.append(cmd)
        if len(run_calls) == 2:
            raise subprocess.CalledProcessError(returncode=13, cmd=cmd, stderr="train failed")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _run)

    code = module.main()

    assert code == 13
    assert len(run_calls) == 2
    assert run_calls[0][:3] == [module.sys.executable, "-m", "pipelines.search.search"]
    assert run_calls[1][:3] == [module.sys.executable, "-m", "pipelines.runners.train"]
    assert completion_messages == ["Experiment execution failed with return code 13"]
