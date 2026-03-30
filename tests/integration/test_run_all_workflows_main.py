"""Integration tests for the master orchestrator `run_all_workflows`.

These tests monkeypatch subprocess execution to verify the high-level
orchestration flow (success and failure paths) without launching real
subprocesses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pipelines.orchestration.master.run_all_workflows as run_all


def test_main_success(monkeypatch: Any) -> None:
    """When all steps succeed, `main()` returns 0 and runs three steps."""

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], text: bool = False, **kwargs: Any) -> SimpleNamespace:
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_all, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(run_all, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(run_all, "log_completion", lambda start_time, msg: None)
    monkeypatch.setattr(run_all.sys, "argv", ["prog"])  # stable parse_args

    rc = run_all.main()
    assert rc == 0
    assert len(calls) == 3


def test_main_fails_on_step(monkeypatch: Any) -> None:
    """When a step fails, `main()` returns that code and reports the failure."""

    state = {"i": 0}

    def fake_run(cmd: list[str], text: bool = False, **kwargs: Any) -> SimpleNamespace:
        state["i"] += 1
        # make the second step fail
        if state["i"] == 2:
            return SimpleNamespace(returncode=5)
        return SimpleNamespace(returncode=0)

    captured: dict[str, str] = {}

    def fake_log_completion(start_time: float, message: str) -> None:
        captured["msg"] = message

    monkeypatch.setattr(run_all, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(run_all, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(run_all, "log_completion", fake_log_completion)
    monkeypatch.setattr(run_all.sys, "argv", ["prog"])  # stable parse_args

    rc = run_all.main()
    assert rc == 5
    assert "failed at step" in captured.get("msg", "")
