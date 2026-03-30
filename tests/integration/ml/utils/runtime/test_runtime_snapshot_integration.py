"""Integration tests for runtime snapshot builders in `ml.utils.runtime.runtime_snapshot`."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import ml.utils.runtime.runtime_snapshot as rs
import pytest
from ml.config.schemas.hardware_cfg import HardwareConfig, HardwareTaskType
from ml.exceptions import RuntimeMLError


def test_find_conda_executable_which(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(rs.shutil, "which", lambda name: str(tmp_path / "conda"))
    assert rs.find_conda_executable() == str(tmp_path / "conda")


def test_find_conda_executable_uses_conda_prefix(monkeypatch: Any, tmp_path: Path) -> None:
    # Simulate no which result, but valid CONDA_PREFIX layout
    monkeypatch.setattr(rs.shutil, "which", lambda name: None)
    conda_prefix = tmp_path / "conda" / "envs" / "myenv"
    conda_prefix.mkdir(parents=True)
    base = conda_prefix.parent.parent
    bin_dir = base / "bin"
    bin_dir.mkdir(parents=True)
    candidate = bin_dir / "conda"
    candidate.write_text("")

    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    monkeypatch.setattr(rs.platform, "system", lambda: "Linux")

    got = rs.find_conda_executable()
    assert Path(got).name == "conda"


def test__run_command_success_and_failure(monkeypatch: Any) -> None:
    # success path
    def fake_run_ok(cmd, check, capture_output, text):
        return types.SimpleNamespace(stdout="ok\n")

    monkeypatch.setattr(rs.subprocess, "run", fake_run_ok)
    assert rs._run_command(["echo", "hi"]) == "ok\n"

    # failure path
    class FakeError(Exception):
        def __init__(self):
            self.stdout = "out"
            self.stderr = "err"
            super().__init__("boom")

    def fake_run_fail(cmd, check, capture_output, text):
        raise FakeError()

    monkeypatch.setattr(rs.subprocess, "run", fake_run_fail)
    with pytest.raises(RuntimeMLError):
        rs._run_command(["false"])


def test_get_conda_env_export_calls_run(monkeypatch: Any) -> None:
    monkeypatch.setattr(rs, "find_conda_executable", lambda: "/usr/bin/conda")
    monkeypatch.setattr(rs, "_run_command", lambda cmd: "env: yaml")
    assert rs.get_conda_env_export() == "env: yaml"


def test_build_runtime_snapshot_happy_and_missing_conda(monkeypatch: Any) -> None:
    monkeypatch.setattr(rs, "get_git_commit", lambda p: "abc123")
    monkeypatch.setattr(rs, "get_runtime_info", lambda: {"python_version": "3.10"})
    monkeypatch.setattr(rs, "get_gpu_info", lambda hw: {"gpu_count": 0})
    monkeypatch.setattr(rs.time, "perf_counter", lambda: 200.0)

    hw = HardwareConfig(task_type=HardwareTaskType.CPU, devices=[])

    # Case A: conda available
    monkeypatch.setattr(rs, "get_conda_env_export", lambda: "name: test\n")
    monkeypatch.setattr(rs, "hash_environment", lambda s: "deadbeef")

    payload = rs.build_runtime_snapshot("2026-03-30T12-00-00", hw, start_time=100.0)
    assert payload["execution"]["git_commit"] == "abc123"
    assert payload["environment"]["conda_env_hash"] == "deadbeef"

    # Case B: conda export fails -> values set to 'Unavailable'
    def raise_exc():
        raise RuntimeError("no conda")

    monkeypatch.setattr(rs, "get_conda_env_export", raise_exc)
    payload2 = rs.build_runtime_snapshot("2026-03-30T12-00-00", hw, start_time=100.0)
    assert payload2["environment"]["conda_env_export"] == "Unavailable"
    assert payload2["environment"]["conda_env_hash"] == "Unavailable"
