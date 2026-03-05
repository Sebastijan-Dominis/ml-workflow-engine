"""Unit tests for conda discovery and command helpers in runtime_snapshot."""

import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest
from ml.exceptions import RuntimeMLError

# Ensure importing runtime_snapshot helpers does not require NVML at test import time.
if "pynvml" not in sys.modules:
    pynvml_stub = types.ModuleType("pynvml")
    pynvml_stub.__dict__.update(
        {
            "NVMLError": Exception,
            "nvmlInit": lambda: None,
            "nvmlDeviceGetCount": lambda: 0,
            "nvmlShutdown": lambda: None,
        }
    )
    sys.modules["pynvml"] = pynvml_stub

runtime_snapshot_module = importlib.import_module("ml.utils.runtime.runtime_snapshot")
find_conda_executable = runtime_snapshot_module.find_conda_executable
get_conda_env_export = runtime_snapshot_module.get_conda_env_export
run_command = runtime_snapshot_module._run_command


pytestmark = pytest.mark.unit


def test_find_conda_executable_prefers_shutil_which(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `find_conda_executable` prefers `shutil.which` results."""
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.shutil.which", lambda _: "/usr/bin/conda")

    result = find_conda_executable()

    assert result == "/usr/bin/conda"


def test_find_conda_executable_resolves_from_conda_prefix_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Linux fallback resolution from `CONDA_PREFIX` when not on `PATH`."""
    # /tmp/.../conda_root/envs/dev -> base resolves to /tmp/.../conda_root
    conda_prefix = tmp_path / "conda_root" / "envs" / "dev"
    candidate = tmp_path / "conda_root" / "bin" / "conda"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.shutil.which", lambda _: None)
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.platform.system", lambda: "Linux")
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))

    result = find_conda_executable()

    assert result == str(candidate)


def test_find_conda_executable_raises_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `RuntimeMLError` is raised when conda cannot be located."""
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.shutil.which", lambda _: None)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    with pytest.raises(RuntimeMLError, match="Could not locate conda executable"):
        find_conda_executable()


def test_run_command_returns_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `_run_command` returns captured standard output."""
    class _Result:
        stdout = "ok\n"

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.subprocess.run", lambda *args, **kwargs: _Result())

    result = run_command(["conda", "--version"])

    assert result == "ok\n"


def test_run_command_wraps_failures_in_runtime_ml_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that subprocess failures are wrapped as `RuntimeMLError`."""
    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["conda"], output="out", stderr="err")

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.subprocess.run", _raise)

    with pytest.raises(RuntimeMLError, match="Command failed"):
        run_command(["conda", "env", "export"])


def test_get_conda_env_export_calls_conda_export_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `get_conda_env_export` executes the expected conda export command."""
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.find_conda_executable", lambda: "/usr/bin/conda")

    calls: list[list[str]] = []

    def _capture(cmd: list[str]) -> str:
        calls.append(cmd)
        return "name: test\n"

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot._run_command", _capture)

    result = get_conda_env_export()

    assert result == "name: test\n"
    assert calls == [["/usr/bin/conda", "env", "export", "--no-builds"]]


def test_get_conda_env_export_wraps_internal_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that internal export failures are wrapped as `RuntimeMLError`."""
    def _raise() -> str:
        raise RuntimeMLError("conda missing")

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.find_conda_executable", _raise)

    with pytest.raises(RuntimeMLError, match="Failed to export conda environment"):
        get_conda_env_export()
