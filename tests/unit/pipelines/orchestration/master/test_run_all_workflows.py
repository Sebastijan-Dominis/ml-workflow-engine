"""Unit tests for master workflow orchestration control flow."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pipelines.orchestration.master import run_all_workflows

pytestmark = pytest.mark.unit


def test_parse_args_uses_documented_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse CLI args with expected defaults for env, logging, owner, and skip flag."""
    monkeypatch.setattr(run_all_workflows.sys, "argv", ["run_all_workflows"])

    args = run_all_workflows.parse_args()

    assert args.env == "dev"
    assert args.logging_level == "INFO"
    assert args.owner == "Sebastijan"
    assert args.skip_if_existing is True


def test_parse_args_converts_skip_flag_from_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert string CLI boolean for `--skip-if-existing` through shared parser."""
    monkeypatch.setattr(
        run_all_workflows.sys,
        "argv",
        [
            "run_all_workflows",
            "--env",
            "test",
            "--logging-level",
            "DEBUG",
            "--owner",
            "CI",
            "--skip-if-existing",
            "false",
        ],
    )

    args = run_all_workflows.parse_args()

    assert args.env == "test"
    assert args.logging_level == "DEBUG"
    assert args.owner == "CI"
    assert args.skip_if_existing is False


def test_run_step_returns_error_code_when_subprocess_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return failing subprocess return code so caller can abort orchestration."""
    monkeypatch.setattr(
        run_all_workflows.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=9),
    )

    code = run_all_workflows.run_step(["python", "-m", "dummy"], "Dummy Step")

    assert code == 9


def test_main_stops_at_first_failed_step_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short-circuit orchestration when a step fails and report the failing step name."""
    setup_calls: list[Path] = []
    run_calls: list[tuple[list[str], str]] = []
    completion_messages: list[str] = []

    monkeypatch.setattr(
        run_all_workflows,
        "parse_args",
        lambda: Namespace(env="dev", logging_level="INFO", owner="owner", skip_if_existing=True),
    )
    monkeypatch.setattr(run_all_workflows, "iso_no_colon", lambda _dt: "20260306T120000")
    monkeypatch.setattr(run_all_workflows, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))
    monkeypatch.setattr(run_all_workflows, "setup_logging", lambda path: setup_calls.append(path))

    def _fake_run_step(cmd: list[str], step_name: str) -> int:
        run_calls.append((cmd, step_name))
        return 2 if step_name == "Freeze All Feature Sets" else 0

    monkeypatch.setattr(run_all_workflows, "run_step", _fake_run_step)
    monkeypatch.setattr(
        run_all_workflows,
        "log_completion",
        lambda _start, message: completion_messages.append(message),
    )

    code = run_all_workflows.main()

    assert code == 2
    assert setup_calls == [
        Path("orchestration_logs/run_all_workflows/20260306T120000_abcdef01/run_all_workflows.log")
    ]
    assert [step_name for _, step_name in run_calls] == [
        "Execute All Data Preprocessing",
        "Freeze All Feature Sets",
    ]
    assert completion_messages == ["Run all workflows failed at step: Freeze All Feature Sets"]


def test_main_runs_all_steps_and_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute all configured steps and emit successful completion message when none fail."""
    run_calls: list[tuple[list[str], str]] = []
    completion_messages: list[str] = []

    monkeypatch.setattr(
        run_all_workflows,
        "parse_args",
        lambda: Namespace(env="prod", logging_level="WARNING", owner="owner", skip_if_existing=False),
    )
    monkeypatch.setattr(run_all_workflows, "iso_no_colon", lambda _dt: "20260306T121500")
    monkeypatch.setattr(run_all_workflows, "uuid4", lambda: SimpleNamespace(hex="1122334455667788"))
    monkeypatch.setattr(run_all_workflows, "setup_logging", lambda _path: None)

    def _fake_run_step(cmd: list[str], step_name: str) -> int:
        run_calls.append((cmd, step_name))
        return 0

    monkeypatch.setattr(run_all_workflows, "run_step", _fake_run_step)
    monkeypatch.setattr(
        run_all_workflows,
        "log_completion",
        lambda _start, message: completion_messages.append(message),
    )

    code = run_all_workflows.main()

    assert code == 0
    assert [step_name for _, step_name in run_calls] == [
        "Execute All Data Preprocessing",
        "Freeze All Feature Sets",
        "Execute All Experiments",
    ]
    assert completion_messages == ["Run all workflows completed successfully."]
    assert run_calls[0][0][-2:] == ["--skip-if-existing", "False"]
    assert run_calls[2][0][2:4] == ["pipelines.orchestration.experiments.execute_all_experiments_with_latest", "--env"]
