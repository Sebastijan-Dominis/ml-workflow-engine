"""Unit tests for batch data-preprocessing orchestration."""

from __future__ import annotations

import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipelines.orchestration.data import execute_all_data_preprocessing as module

pytestmark = pytest.mark.unit


def test_parse_args_defaults_to_skip_existing_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse CLI defaults and keep idempotent skip behavior enabled."""
    monkeypatch.setattr(module.sys, "argv", ["execute_all_data_preprocessing"])

    args = module.parse_args()

    assert args.skip_if_existing is True


def test_parse_args_converts_skip_existing_from_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert text boolean values for `--skip-if-existing` via shared parser."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["execute_all_data_preprocessing", "--skip-if-existing", "false"],
    )

    args = module.parse_args()

    assert args.skip_if_existing is False


def test_run_cmd_invokes_subprocess_with_expected_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call subprocess with strict failure handling and captured text output."""
    captured: dict[str, Any] = {}

    def _fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    module.run_cmd(["python", "-m", "pipelines.data.build_interim_dataset"])

    assert captured["cmd"] == ["python", "-m", "pipelines.data.build_interim_dataset"]
    assert captured["kwargs"] == {"check": True, "capture_output": True, "text": True}


def test_run_cmd_logs_stdout_when_present(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit subprocess standard output into orchestration logs for observability."""

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="command output"),
    )

    with caplog.at_level("INFO", logger=module.__name__):
        module.run_cmd(["python", "-m", "pipelines.data.build_processed_dataset"])

    assert "command output" in caplog.text


def test_main_successfully_completes_when_no_discoverable_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return success when none of the data/config roots exist yet."""
    monkeypatch.chdir(tmp_path)

    setup_paths: list[Path] = []
    completion_messages: list[str] = []

    monkeypatch.setattr(module, "parse_args", lambda: Namespace(skip_if_existing=True))
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T140000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="aabbccddeeff0011"))
    monkeypatch.setattr(module, "setup_logging", lambda path: setup_paths.append(path))
    monkeypatch.setattr(
        module,
        "log_completion",
        lambda *, start_time, message: completion_messages.append(message),
    )

    code = module.main()

    assert code == 0
    assert setup_paths == [
        Path("orchestration_logs/data/execute_all_data_preprocessing/20260306T140000_aabbccdd/data_preprocessing.log")
    ]
    assert completion_messages == ["Full data preprocessing run completed successfully."]


def test_main_skips_steps_when_outputs_exist_and_skip_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Skip raw/interim/processed step execution when expected outputs already exist."""
    monkeypatch.chdir(tmp_path)

    raw_snapshot = tmp_path / "data" / "raw" / "hotel_bookings" / "v1" / "snap_001"
    raw_snapshot.mkdir(parents=True)
    (raw_snapshot / "metadata.json").write_text("{}", encoding="utf-8")

    (tmp_path / "configs" / "data" / "interim" / "hotel_bookings").mkdir(parents=True)
    (tmp_path / "configs" / "data" / "interim" / "hotel_bookings" / "v1.yaml").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "data" / "interim" / "hotel_bookings" / "v1" / "run_001").mkdir(parents=True)

    (tmp_path / "configs" / "data" / "processed" / "hotel_bookings").mkdir(parents=True)
    (tmp_path / "configs" / "data" / "processed" / "hotel_bookings" / "v1.yaml").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "data" / "processed" / "hotel_bookings" / "v1" / "run_002").mkdir(parents=True)

    calls: list[list[str]] = []

    monkeypatch.setattr(module, "parse_args", lambda: Namespace(skip_if_existing=True))
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T140500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))
    monkeypatch.setattr(module, "setup_logging", lambda _path: None)
    monkeypatch.setattr(module, "run_cmd", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(module, "log_completion", lambda *, start_time, message: None)

    code = module.main()

    assert code == 0
    assert calls == []


def test_main_runs_all_three_stage_commands_when_not_skipping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run raw registration, interim build, and processed build in order when needed."""
    monkeypatch.chdir(tmp_path)

    (tmp_path / "data" / "raw" / "hotel_bookings" / "v1" / "snap_001").mkdir(parents=True)

    (tmp_path / "configs" / "data" / "interim" / "hotel_bookings").mkdir(parents=True)
    (tmp_path / "configs" / "data" / "interim" / "hotel_bookings" / "v1.yaml").write_text(
        "{}",
        encoding="utf-8",
    )

    (tmp_path / "configs" / "data" / "processed" / "hotel_bookings").mkdir(parents=True)
    (tmp_path / "configs" / "data" / "processed" / "hotel_bookings" / "v2.yaml").write_text(
        "{}",
        encoding="utf-8",
    )

    commands: list[list[str]] = []

    monkeypatch.setattr(module, "parse_args", lambda: Namespace(skip_if_existing=False))
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T141000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="9988776655443322"))
    monkeypatch.setattr(module, "setup_logging", lambda _path: None)
    monkeypatch.setattr(module, "run_cmd", lambda cmd: commands.append(cmd))
    monkeypatch.setattr(module, "log_completion", lambda *, start_time, message: None)

    code = module.main()

    assert code == 0
    assert len(commands) == 3
    assert commands[0][:3] == [module.sys.executable, "-m", "pipelines.data.register_raw_snapshot"]
    assert commands[1][:3] == [module.sys.executable, "-m", "pipelines.data.build_interim_dataset"]
    assert commands[2][:3] == [module.sys.executable, "-m", "pipelines.data.build_processed_dataset"]


def test_main_returns_subprocess_code_and_logs_completion_message_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return delegated command failure code and emit the failing command summary."""
    monkeypatch.chdir(tmp_path)

    (tmp_path / "data" / "raw" / "hotel_bookings" / "v1" / "snap_001").mkdir(parents=True)

    completion_messages: list[str] = []

    monkeypatch.setattr(module, "parse_args", lambda: Namespace(skip_if_existing=False))
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T141500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="deadbeefcafebabe"))
    monkeypatch.setattr(module, "setup_logging", lambda _path: None)

    def _raise(_cmd: list[str]) -> None:
        raise subprocess.CalledProcessError(
            returncode=27,
            cmd=["python", "-m", "pipelines.data.register_raw_snapshot"],
        )

    monkeypatch.setattr(module, "run_cmd", _raise)
    monkeypatch.setattr(
        module,
        "log_completion",
        lambda *, start_time, message: completion_messages.append(message),
    )

    code = module.main()

    assert code == 27
    assert completion_messages == [
        "Command 'python -m pipelines.data.register_raw_snapshot' failed with exit code 27"
    ]


def test_main_returns_one_on_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return generic failure code for unexpected exceptions outside subprocess errors."""
    monkeypatch.chdir(tmp_path)

    (tmp_path / "data" / "raw" / "hotel_bookings" / "v1" / "snap_001").mkdir(parents=True)

    monkeypatch.setattr(module, "parse_args", lambda: Namespace(skip_if_existing=False))
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T142000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="0123456789abcdef"))
    monkeypatch.setattr(module, "setup_logging", lambda _path: None)
    monkeypatch.setattr(module, "run_cmd", lambda _cmd: (_ for _ in ()).throw(RuntimeError("boom")))

    code = module.main()

    assert code == 1


def test_main_ignores_non_directory_entries_during_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ignore non-directory entries across raw/interim/processed discovery roots."""
    monkeypatch.chdir(tmp_path)

    # Raw discovery: non-directory under data/raw, under dataset version root, and under snapshot root.
    raw_root = tmp_path / "data" / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / "README.txt").write_text("x", encoding="utf-8")
    data_dir = raw_root / "hotel_bookings"
    data_dir.mkdir()
    (data_dir / "v1.txt").write_text("x", encoding="utf-8")
    version_dir = data_dir / "v1"
    version_dir.mkdir()
    (version_dir / "not_a_snapshot.txt").write_text("x", encoding="utf-8")

    # Interim/processed discovery: non-directory entries under config roots.
    interim_root = tmp_path / "configs" / "data" / "interim"
    interim_root.mkdir(parents=True)
    (interim_root / "notes.txt").write_text("x", encoding="utf-8")

    processed_root = tmp_path / "configs" / "data" / "processed"
    processed_root.mkdir(parents=True)
    (processed_root / "notes.txt").write_text("x", encoding="utf-8")

    calls: list[list[str]] = []

    monkeypatch.setattr(module, "parse_args", lambda: Namespace(skip_if_existing=False))
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260307T140000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="cafebabedeadbeef"))
    monkeypatch.setattr(module, "setup_logging", lambda _path: None)
    monkeypatch.setattr(module, "run_cmd", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(module, "log_completion", lambda *, start_time, message: None)

    code = module.main()

    assert code == 0
    assert calls == []
