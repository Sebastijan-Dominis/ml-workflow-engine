"""Integration tests for orchestration CLI wiring across multi-step workflows."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipelines.orchestration.experiments import execute_experiment_with_latest
from pipelines.orchestration.master import run_all_workflows

pytestmark = pytest.mark.integration


def test_execute_experiment_with_latest_parses_cli_and_runs_all_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run orchestration with real CLI parsing and verify downstream command chain."""
    monkeypatch.setattr(
        execute_experiment_with_latest.sys,
        "argv",
        [
            "execute_experiment_with_latest",
            "--problem",
            "no_show",
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
            "false",
            "--experiment-id",
            "exp_42",
            "--overwrite-existing",
            "false",
            "--top-k",
            "12",
        ],
    )

    monkeypatch.setattr(execute_experiment_with_latest, "iso_no_colon", lambda _dt: "20260307T100000")
    monkeypatch.setattr(
        execute_experiment_with_latest,
        "uuid4",
        lambda: SimpleNamespace(hex="abcdef0123456789"),
    )

    setup_calls: list[tuple[Path, int]] = []
    completion_messages: list[str] = []
    run_calls: list[tuple[list[str], dict[str, Any]]] = []

    monkeypatch.setattr(
        execute_experiment_with_latest,
        "setup_logging",
        lambda path, level: setup_calls.append((path, level)),
    )
    monkeypatch.setattr(
        execute_experiment_with_latest,
        "log_completion",
        lambda _start, message: completion_messages.append(message),
    )

    def _run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        run_calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(execute_experiment_with_latest.subprocess, "run", _run)

    code = execute_experiment_with_latest.main()

    assert code == 0
    assert len(run_calls) == 4
    assert setup_calls == [
        (
            Path(
                "orchestration_logs/experiments/execute_experiment_with_latest/"
                "20260307T100000_abcdef01/experiment_execution.log"
            ),
            execute_experiment_with_latest.logging.INFO,
        )
    ]

    search_cmd, search_kwargs = run_calls[0]
    assert search_cmd[:3] == [
        execute_experiment_with_latest.sys.executable,
        "-m",
        "pipelines.search.search",
    ]
    assert search_cmd[search_cmd.index("--problem") + 1] == "no_show"
    assert search_cmd[search_cmd.index("--segment") + 1] == "global"
    assert search_cmd[search_cmd.index("--version") + 1] == "v1"
    assert search_cmd[search_cmd.index("--env") + 1] == "test"
    assert search_cmd[search_cmd.index("--strict") + 1] == "True"
    assert search_cmd[search_cmd.index("--owner") + 1] == "CI"
    assert search_cmd[search_cmd.index("--clean-up-failure-management") + 1] == "False"
    assert search_cmd[search_cmd.index("--experiment-id") + 1] == "exp_42"
    assert search_kwargs == {"check": True}

    train_cmd, train_kwargs = run_calls[1]
    assert train_cmd[:3] == [
        execute_experiment_with_latest.sys.executable,
        "-m",
        "pipelines.runners.train",
    ]
    assert train_kwargs == {"check": True, "capture_output": True, "text": True, "encoding": "utf-8"}

    evaluate_cmd, evaluate_kwargs = run_calls[2]
    assert evaluate_cmd[:3] == [
        execute_experiment_with_latest.sys.executable,
        "-m",
        "pipelines.runners.evaluate",
    ]
    assert evaluate_kwargs == {
        "check": True,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
    }

    explain_cmd, explain_kwargs = run_calls[3]
    assert explain_cmd[:3] == [
        execute_experiment_with_latest.sys.executable,
        "-m",
        "pipelines.runners.explain",
    ]
    assert explain_cmd[explain_cmd.index("--top-k") + 1] == "12"
    assert explain_kwargs == {"check": True, "capture_output": True, "text": True, "encoding": "utf-8"}

    assert completion_messages == [
        "Experiment execution completed successfully for problem=no_show, segment=global, version=v1"
    ]


def test_run_all_workflows_parses_cli_and_stops_after_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run master orchestration with real CLI parsing and verify fail-fast behavior."""
    monkeypatch.setattr(
        run_all_workflows.sys,
        "argv",
        [
            "run_all_workflows",
            "--env",
            "test",
            "--logging-level",
            "WARNING",
            "--owner",
            "CI",
            "--skip-if-existing",
            "false",
        ],
    )

    monkeypatch.setattr(run_all_workflows, "iso_no_colon", lambda _dt: "20260307T101500")
    monkeypatch.setattr(run_all_workflows, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))

    setup_calls: list[Path] = []
    completion_messages: list[str] = []
    seen_cmds: list[list[str]] = []

    monkeypatch.setattr(run_all_workflows, "setup_logging", lambda path: setup_calls.append(path))
    monkeypatch.setattr(run_all_workflows, "log_completion", lambda _start, message: completion_messages.append(message))

    def _run(cmd: list[str], *, text: bool) -> SimpleNamespace:
        assert text is True
        seen_cmds.append(cmd)
        if "pipelines.orchestration.features.freeze_all_feature_sets" in cmd:
            return SimpleNamespace(returncode=7)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_all_workflows.subprocess, "run", _run)

    code = run_all_workflows.main()

    assert code == 7
    assert setup_calls == [
        Path("orchestration_logs/run_all_workflows/20260307T101500_00112233/run_all_workflows.log")
    ]
    assert len(seen_cmds) == 2
    assert seen_cmds[0][:3] == [
        run_all_workflows.sys.executable,
        "-m",
        "pipelines.orchestration.data.execute_all_data_preprocessing",
    ]
    assert seen_cmds[0][-2:] == ["--skip-if-existing", "False"]
    assert seen_cmds[1][:3] == [
        run_all_workflows.sys.executable,
        "-m",
        "pipelines.orchestration.features.freeze_all_feature_sets",
    ]
    assert completion_messages == ["Run all workflows failed at step: Freeze All Feature Sets"]
