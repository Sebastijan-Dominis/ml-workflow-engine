"""Unit tests for single-model experiment orchestration flow."""

from __future__ import annotations

import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipelines.orchestration.experiments import execute_experiment_with_latest as module

pytestmark = pytest.mark.unit


def test_parse_args_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse default optional flags for strictness, logging, ownership, and IDs."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "execute_experiment_with_latest",
            "--problem",
            "no_show",
            "--segment",
            "global",
            "--version",
            "v1",
        ],
    )

    args = module.parse_args()

    assert args.problem == "no_show"
    assert args.segment == "global"
    assert args.version == "v1"
    assert args.env == "default"
    assert args.strict is True
    assert args.logging_level == "INFO"
    assert args.owner == "Sebastijan"
    assert args.clean_up_failure_management is True
    assert args.experiment_id is None
    assert args.overwrite_existing is False
    assert args.top_k is None


def test_parse_args_converts_boolean_and_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert text booleans and parse optional override flags correctly."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "execute_experiment_with_latest",
            "--problem",
            "adr",
            "--segment",
            "city_hotel",
            "--version",
            "v2",
            "--env",
            "prod",
            "--strict",
            "false",
            "--logging-level",
            "DEBUG",
            "--owner",
            "CI",
            "--clean-up-failure-management",
            "false",
            "--experiment-id",
            "20260306T170000_abcd1234",
            "--overwrite-existing",
            "true",
            "--top-k",
            "25",
        ],
    )

    args = module.parse_args()

    assert args.problem == "adr"
    assert args.segment == "city_hotel"
    assert args.version == "v2"
    assert args.env == "prod"
    assert args.strict is False
    assert args.logging_level == "DEBUG"
    assert args.owner == "CI"
    assert args.clean_up_failure_management is False
    assert args.experiment_id == "20260306T170000_abcd1234"
    assert args.overwrite_existing is True
    assert args.top_k == 25


def test_parse_args_rejects_invalid_env_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported environment values through argparse choices."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "execute_experiment_with_latest",
            "--problem",
            "adr",
            "--segment",
            "city_hotel",
            "--version",
            "v1",
            "--env",
            "qa",
        ],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_main_runs_search_train_evaluate_explain_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute all orchestration stages with expected command payloads and options."""
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            env="test",
            strict=True,
            logging_level="INFO",
            owner="Owner",
            clean_up_failure_management=True,
            experiment_id="20260306T171000_deadbeef",
            overwrite_existing=False,
            top_k=15,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T171100")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))

    setup_calls: list[tuple[Path, int]] = []
    run_calls: list[tuple[list[str], dict[str, Any]]] = []
    completion_messages: list[str] = []

    monkeypatch.setattr(module, "setup_logging", lambda path, level: setup_calls.append((path, level)))

    def _fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        run_calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module, "log_completion", lambda start, message: completion_messages.append(message))

    code = module.main()

    assert code == 0
    assert setup_calls == [
        (
            Path(
                "orchestration_logs/experiments/execute_experiment_with_latest/"
                "20260306T171100_abcdef01/experiment_execution.log"
            ),
            module.logging.INFO,
        )
    ]

    assert len(run_calls) == 4

    search_cmd, search_kwargs = run_calls[0]
    assert search_cmd[:3] == [module.sys.executable, "-m", "pipelines.search.search"]
    assert search_cmd[search_cmd.index("--problem") + 1] == "no_show"
    assert search_cmd[search_cmd.index("--segment") + 1] == "global"
    assert search_cmd[search_cmd.index("--version") + 1] == "v1"
    assert search_cmd[search_cmd.index("--env") + 1] == "test"
    assert search_cmd[search_cmd.index("--strict") + 1] == "True"
    assert search_cmd[search_cmd.index("--owner") + 1] == "Owner"
    assert search_cmd[search_cmd.index("--clean-up-failure-management") + 1] == "True"
    assert search_cmd[search_cmd.index("--overwrite-existing") + 1] == "False"
    assert search_cmd[search_cmd.index("--experiment-id") + 1] == "20260306T171000_deadbeef"
    assert search_kwargs == {"check": True}

    train_cmd, train_kwargs = run_calls[1]
    assert train_cmd[:3] == [module.sys.executable, "-m", "pipelines.runners.train"]
    assert train_kwargs == {"check": True, "capture_output": True, "text": True, "encoding": "utf-8"}

    eval_cmd, eval_kwargs = run_calls[2]
    assert eval_cmd[:3] == [module.sys.executable, "-m", "pipelines.runners.evaluate"]
    assert eval_kwargs == {"check": True, "capture_output": True, "text": True, "encoding": "utf-8"}

    explain_cmd, explain_kwargs = run_calls[3]
    assert explain_cmd[:3] == [module.sys.executable, "-m", "pipelines.runners.explain"]
    assert explain_cmd[explain_cmd.index("--top-k") + 1] == "15"
    assert explain_kwargs == {"check": True, "capture_output": True, "text": True, "encoding": "utf-8"}

    assert completion_messages == [
        "Experiment execution completed successfully for problem=no_show, segment=global, version=v1"
    ]


def test_main_omits_optional_flags_when_not_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not inject `--experiment-id` or `--top-k` when optional values are missing."""
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="adr",
            segment="city_hotel",
            version="v2",
            env="dev",
            strict=False,
            logging_level="DEBUG",
            owner="Owner",
            clean_up_failure_management=False,
            experiment_id=None,
            overwrite_existing=True,
            top_k=None,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T171500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    run_calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        run_calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module, "log_completion", lambda start, message: None)

    code = module.main()

    assert code == 0
    search_cmd = run_calls[0]
    explain_cmd = run_calls[3]
    assert "--experiment-id" not in search_cmd
    assert "--top-k" not in explain_cmd


def test_main_returns_subprocess_return_code_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return failing subprocess code and stop executing remaining stages."""
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="cancellation",
            segment="global",
            version="v1",
            env="default",
            strict=True,
            logging_level="INFO",
            owner="Owner",
            clean_up_failure_management=True,
            experiment_id=None,
            overwrite_existing=False,
            top_k=None,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T172000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="8899aabbccddeeff"))
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    run_calls: list[list[str]] = []
    completion_messages: list[str] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        run_calls.append(cmd)
        if len(run_calls) == 2:
            raise subprocess.CalledProcessError(
                returncode=23,
                cmd=cmd,
                stderr="first line\nlong traceback",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module, "log_completion", lambda start, message: completion_messages.append(message))

    code = module.main()

    assert code == 23
    assert len(run_calls) == 2
    assert run_calls[0][:3] == [module.sys.executable, "-m", "pipelines.search.search"]
    assert run_calls[1][:3] == [module.sys.executable, "-m", "pipelines.runners.train"]
    assert completion_messages == ["Experiment execution failed with return code 23"]


def test_main_falls_back_to_info_when_logging_level_is_unrecognized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use INFO fallback when logging level string does not map to logging constants."""
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="adr",
            segment="city_hotel",
            version="v2",
            env="dev",
            strict=True,
            logging_level="not-a-level",
            owner="Owner",
            clean_up_failure_management=True,
            experiment_id=None,
            overwrite_existing=False,
            top_k=None,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T173000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="a1b2c3d4e5f60708"))

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        module,
        "setup_logging",
        lambda path, level: captured.update({"path": path, "level": level}),
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    monkeypatch.setattr(module, "log_completion", lambda _start, _message: None)

    code = module.main()

    assert code == 0
    assert captured["level"] == module.logging.INFO
    assert str(captured["path"]).endswith("experiment_execution.log")


def test_main_handles_subprocess_failure_without_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return failing code and completion message even when CalledProcessError lacks stderr."""
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="cancellation",
            segment="global",
            version="v1",
            env="default",
            strict=True,
            logging_level="INFO",
            owner="Owner",
            clean_up_failure_management=True,
            experiment_id=None,
            overwrite_existing=False,
            top_k=None,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T173500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="1029384756abcdef"))
    monkeypatch.setattr(module, "setup_logging", lambda path, level: None)

    completion_messages: list[str] = []

    def _fail_without_stderr(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        _ = kwargs
        raise subprocess.CalledProcessError(returncode=11, cmd=cmd, stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fail_without_stderr)
    monkeypatch.setattr(module, "log_completion", lambda _start, message: completion_messages.append(message))

    code = module.main()

    assert code == 11
    assert completion_messages == ["Experiment execution failed with return code 11"]
