"""Unit tests for experiment-batch orchestration with latest artifacts."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipelines.orchestration.experiments import execute_all_experiments_with_latest as module

pytestmark = pytest.mark.unit


def test_parse_args_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse documented default values for environment and orchestration flags."""
    monkeypatch.setattr(module.sys, "argv", ["execute_all_experiments_with_latest"])

    args = module.parse_args()

    assert args.env == "dev"
    assert args.strict is True
    assert args.logging_level == "INFO"
    assert args.owner == "Sebastijan"
    assert args.clean_up_failure_management is True
    assert args.overwrite_existing is False
    assert args.top_k is None
    assert args.skip_if_existing is True


def test_discover_models_collects_problem_segment_version_tuples(tmp_path: Path) -> None:
    """Discover valid model specs while ignoring non-directory and non-YAML entries."""
    specs_root = tmp_path / "model_specs"
    (specs_root / "cancellation" / "city_hotel").mkdir(parents=True)
    (specs_root / "cancellation" / "city_hotel" / "v1.yaml").write_text("x: 1", encoding="utf-8")
    (specs_root / "cancellation" / "city_hotel" / "notes.txt").write_text("ignore", encoding="utf-8")
    (specs_root / "README.md").write_text("ignore", encoding="utf-8")

    original_dir = module.MODEL_SPECS_DIR
    module.MODEL_SPECS_DIR = specs_root
    try:
        result = module.discover_models()
    finally:
        module.MODEL_SPECS_DIR = original_dir

    assert result == [("cancellation", "city_hotel", "v1")]


def test_discover_models_ignores_non_directory_segment_entries(tmp_path: Path) -> None:
    """Ignore non-directory entries under problem folders during model discovery."""
    specs_root = tmp_path / "model_specs"
    problem_dir = specs_root / "cancellation"
    segment_dir = problem_dir / "city_hotel"
    segment_dir.mkdir(parents=True)
    (problem_dir / "README.txt").write_text("ignore", encoding="utf-8")
    (segment_dir / "v1.yaml").write_text("x: 1", encoding="utf-8")

    original_dir = module.MODEL_SPECS_DIR
    module.MODEL_SPECS_DIR = specs_root
    try:
        result = module.discover_models()
    finally:
        module.MODEL_SPECS_DIR = original_dir

    assert result == [("cancellation", "city_hotel", "v1")]


def test_discover_models_returns_empty_when_specs_directory_missing(tmp_path: Path) -> None:
    """Return no models when specs root is absent instead of raising filesystem errors."""
    missing_specs_root = tmp_path / "does_not_exist"

    original_dir = module.MODEL_SPECS_DIR
    module.MODEL_SPECS_DIR = missing_specs_root
    try:
        result = module.discover_models()
    finally:
        module.MODEL_SPECS_DIR = original_dir

    assert result == []


def test_run_model_skips_when_existing_experiments_and_skip_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Short-circuit model run when experiment folders already exist and skipping is enabled."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments" / "adr" / "city_hotel" / "v1" / "existing_run").mkdir(parents=True)

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    code = module.run_model(
        "adr",
        "city_hotel",
        "v1",
        args=Namespace(
            env="dev",
            strict=True,
            logging_level="INFO",
            owner="owner",
            clean_up_failure_management=True,
            overwrite_existing=False,
            top_k=None,
            skip_if_existing=True,
        ),
        start_time=0.0,
    )

    assert code == 0
    assert calls == []


def test_run_model_builds_command_and_passes_top_k_when_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Delegate to downstream orchestration module with correctly mapped CLI arguments."""
    monkeypatch.chdir(tmp_path)

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module, "log_completion", lambda _start, _message: None)

    code = module.run_model(
        "adr",
        "city_hotel",
        "v2",
        args=Namespace(
            env="prod",
            strict=False,
            logging_level="warning",
            owner="CI",
            clean_up_failure_management=False,
            overwrite_existing=True,
            top_k=20,
            skip_if_existing=False,
        ),
        start_time=1.23,
    )

    assert code == 0
    cmd = calls[0]
    assert cmd[:3] == [module.sys.executable, "-m", "pipelines.orchestration.experiments.execute_experiment_with_latest"]
    assert "--problem" in cmd
    assert "--segment" in cmd
    assert "--version" in cmd
    assert cmd[cmd.index("--env") + 1] == "prod"
    assert cmd[cmd.index("--strict") + 1] == "False"
    assert cmd[cmd.index("--logging-level") + 1] == "WARNING"
    assert cmd[cmd.index("--owner") + 1] == "CI"
    assert cmd[cmd.index("--clean-up-failure-management") + 1] == "False"
    assert cmd[cmd.index("--overwrite-existing") + 1] == "True"
    assert cmd[cmd.index("--top-k") + 1] == "20"


def test_run_model_logs_failure_completion_and_returns_subprocess_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Log shared-run failure completion message and return delegated non-zero code."""
    monkeypatch.chdir(tmp_path)

    completion_messages: list[str] = []

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda _cmd, **_kwargs: SimpleNamespace(returncode=31, stdout="", stderr="boom"),
    )
    monkeypatch.setattr(module, "log_completion", lambda _start, message: completion_messages.append(message))

    code = module.run_model(
        "adr",
        "city_hotel",
        "v9",
        args=Namespace(
            env="dev",
            strict=True,
            logging_level="INFO",
            owner="owner",
            clean_up_failure_management=True,
            overwrite_existing=False,
            top_k=None,
            skip_if_existing=False,
        ),
        start_time=12.34,
    )

    assert code == 31
    assert completion_messages == [
        "Experiments run failed with model problem=adr, segment=city_hotel, version=v9 with return code 31"
    ]


def test_main_returns_zero_when_no_models_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complete successfully when no model specs are present for execution."""
    completion_messages: list[str] = []

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            env="default",
            strict=True,
            logging_level="INFO",
            owner="owner",
            clean_up_failure_management=True,
            overwrite_existing=False,
            top_k=None,
            skip_if_existing=True,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T132000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))
    monkeypatch.setattr(module, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(module, "discover_models", lambda: [])
    monkeypatch.setattr(
        module,
        "log_completion",
        lambda _start, message: completion_messages.append(message),
    )

    code = module.main()

    assert code == 0
    assert completion_messages == ["Experiment execution completed successfully"]


def test_main_continues_after_failures_and_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue processing all models and return non-zero if any delegated run fails."""
    completion_messages: list[str] = []
    run_calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            env="dev",
            strict=True,
            logging_level="INFO",
            owner="owner",
            clean_up_failure_management=True,
            overwrite_existing=False,
            top_k=None,
            skip_if_existing=False,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T132500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="8899aabbccddeeff"))
    monkeypatch.setattr(module, "setup_logging", lambda **kwargs: None)

    models = [("adr", "city_hotel", "v1"), ("cancellation", "resort_hotel", "v3")]
    monkeypatch.setattr(module, "discover_models", lambda: models)

    def _fake_run_model(
        problem: str,
        segment: str,
        version: str,
        *,
        args: Namespace,
        start_time: float,
    ) -> int:
        run_calls.append((problem, segment, version))
        return 9 if problem == "cancellation" else 0

    monkeypatch.setattr(module, "run_model", _fake_run_model)
    monkeypatch.setattr(
        module,
        "log_completion",
        lambda _start, message: completion_messages.append(message),
    )

    code = module.main()

    assert code == 1
    assert run_calls == models
    assert completion_messages == ["Experiment execution completed with some failures"]


def test_main_returns_zero_when_all_models_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return zero and report successful completion when every delegated model run succeeds."""
    completion_messages: list[str] = []
    run_calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            env="prod",
            strict=False,
            logging_level="WARNING",
            owner="owner",
            clean_up_failure_management=False,
            overwrite_existing=True,
            top_k=15,
            skip_if_existing=False,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T133000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="cafebabedeadbeef"))
    monkeypatch.setattr(module, "setup_logging", lambda **kwargs: None)

    models = [("adr", "city_hotel", "v1"), ("no_show", "global", "v2")]
    monkeypatch.setattr(module, "discover_models", lambda: models)

    def _fake_run_model(
        problem: str,
        segment: str,
        version: str,
        *,
        args: Namespace,
        start_time: float,
    ) -> int:
        _ = args, start_time
        run_calls.append((problem, segment, version))
        return 0

    monkeypatch.setattr(module, "run_model", _fake_run_model)
    monkeypatch.setattr(module, "log_completion", lambda _start, message: completion_messages.append(message))

    code = module.main()

    assert code == 0
    assert run_calls == models
    assert completion_messages == ["Experiment execution completed successfully"]
