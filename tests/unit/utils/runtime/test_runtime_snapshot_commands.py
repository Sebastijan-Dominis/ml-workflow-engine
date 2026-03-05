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
    """Test that find_conda_executable returns the path from shutil.which if it is found. The test uses monkeypatch to replace shutil.which with a fake function that returns a specific path (e.g., "/usr/bin/conda"), then calls find_conda_executable and asserts that the returned path matches the expected value from the fake shutil.which, confirming that find_conda_executable correctly uses shutil.which to locate the conda executable when it is available."""
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.shutil.which", lambda _: "/usr/bin/conda")

    result = find_conda_executable()

    assert result == "/usr/bin/conda"


def test_find_conda_executable_resolves_from_conda_prefix_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that find_conda_executable correctly constructs the path to the conda executable based on the CONDA_PREFIX environment variable when running on Linux and shutil.which does not find conda. The test sets up a fake CONDA_PREFIX path that simulates a typical conda environment structure, uses monkeypatch to replace shutil.which with a function that returns None (indicating conda is not found in PATH) and platform.system with a function that returns "Linux", then calls find_conda_executable and asserts that the returned path matches the expected path constructed from CONDA_PREFIX, confirming that find_conda_executable correctly falls back to constructing the path from CONDA_PREFIX on Linux when conda is not found in PATH."""
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
    """Test that find_conda_executable raises a RuntimeMLError when the conda executable cannot be found by either shutil.which or by resolving from CONDA_PREFIX. The test uses monkeypatch to replace shutil.which with a function that returns None (indicating conda is not found in PATH) and deletes the CONDA_PREFIX environment variable, then calls find_conda_executable and asserts that a RuntimeMLError is raised with a message indicating that the conda executable could not be located, confirming that find_conda_executable correctly handles the case where conda cannot be found and raises an appropriate error."""
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.shutil.which", lambda _: None)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    with pytest.raises(RuntimeMLError, match="Could not locate conda executable"):
        find_conda_executable()


def test_run_command_returns_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _run_command executes the given command and returns its standard output. The test uses monkeypatch to replace subprocess.run with a fake function that returns an object with a stdout attribute containing a specific string (e.g., "ok\n"), then calls _run_command with a sample command and asserts that the returned value matches the expected stdout string, confirming that _run_command correctly captures and returns the standard output of the executed command."""
    class _Result:
        stdout = "ok\n"

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.subprocess.run", lambda *args, **kwargs: _Result())

    result = run_command(["conda", "--version"])

    assert result == "ok\n"


def test_run_command_wraps_failures_in_runtime_ml_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that if the subprocess call within _run_command raises a CalledProcessError, _run_command catches this and raises a RuntimeMLError with an appropriate error message that includes the original exception message. The test uses monkeypatch to replace subprocess.run with a fake function that raises a CalledProcessError, then calls _run_command and asserts that a RuntimeMLError is raised with a message indicating that the command failed and that the original CalledProcessError message is included, confirming that _run_command correctly handles errors from subprocess calls and raises an appropriate RuntimeMLError."""
    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["conda"], output="out", stderr="err")

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.subprocess.run", _raise)

    with pytest.raises(RuntimeMLError, match="Command failed"):
        run_command(["conda", "env", "export"])


def test_get_conda_env_export_calls_conda_export_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_conda_env_export calls the conda command to export the environment and returns its output. The test uses monkeypatch to replace find_conda_executable with a function that returns a specific path to the conda executable, and replaces _run_command with a function that captures the command it is called with and returns a specific string (e.g., "name: test\n"), then calls get_conda_env_export and asserts that the returned value matches the expected output string and that _run_command was called with the expected command to export the conda environment, confirming that get_conda_env_export correctly constructs and executes the conda command to export the environment and returns its output."""
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
    """Test that if find_conda_executable raises a RuntimeMLError when get_conda_env_export tries to locate the conda executable, get_conda_env_export catches this and raises a RuntimeMLError with an appropriate error message that includes the original exception message. The test uses monkeypatch to replace find_conda_executable with a function that raises a RuntimeMLError, then calls get_conda_env_export and asserts that a RuntimeMLError is raised with a message indicating a failure to export the conda environment and that the original RuntimeMLError message is included, confirming that get_conda_env_export correctly handles errors from find_conda_executable and raises an appropriate RuntimeMLError."""
    def _raise() -> str:
        raise RuntimeMLError("conda missing")

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.find_conda_executable", _raise)

    with pytest.raises(RuntimeMLError, match="Failed to export conda environment"):
        get_conda_env_export()
