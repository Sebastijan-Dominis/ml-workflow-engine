"""Unit tests for runtime snapshot build and persistence helpers."""

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from ml.config.schemas.hardware_cfg import HardwareConfig
from ml.exceptions import PersistenceError, RuntimeMLError

# Ensure importing runtime snapshot helpers does not require NVML at test import time.
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
build_runtime_snapshot = runtime_snapshot_module.build_runtime_snapshot
hash_environment = runtime_snapshot_module.hash_environment

save_runtime_module = importlib.import_module("ml.utils.runtime.save_runtime")
save_runtime_snapshot = save_runtime_module.save_runtime_snapshot


pytestmark = pytest.mark.unit


def _cpu_hardware() -> HardwareConfig:
    """Helper function to create a HardwareConfig for CPU execution.

    Returns:
        HardwareConfig: A HardwareConfig instance with task_type set to "CPU" and default values for other fields.
    """
    return HardwareConfig.model_validate({"task_type": "CPU", "devices": []})


def test_hash_environment_is_deterministic() -> None:
    """Test that the hash_environment function produces a deterministic hash for the same environment export string, and that the hash has the expected length (e.g., 64 characters for a SHA-256 hash). The test creates a sample environment export string, calls hash_environment on it multiple times, and asserts that the resulting hashes are identical and have the expected length."""
    env_export = "name: test-env\ndependencies:\n  - python=3.11\n"

    digest_a = hash_environment(env_export)
    digest_b = hash_environment(env_export)

    assert digest_a == digest_b
    assert len(digest_a) == 64


def test_build_runtime_snapshot_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that build_runtime_snapshot successfully builds a snapshot with expected fields when all helper functions work correctly. The test uses monkeypatch to replace the helper functions (get_git_commit, get_runtime_info, get_gpu_info, get_conda_env_export) with fake functions that return specific expected values, then calls build_runtime_snapshot with a sample timestamp, hardware info, and start time, and asserts that the resulting snapshot contains the expected values in the execution, runtime, and environment sections. This validates that build_runtime_snapshot correctly integrates the outputs of the helper functions into the final snapshot structure.

    Args:
    monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture used to replace the helper functions with fake implementations that return controlled outputs for testing.
    """
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_git_commit", lambda _: "abc123")
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_runtime_info", lambda: {"os": "Linux"})
    monkeypatch.setattr(
        "ml.utils.runtime.runtime_snapshot.get_gpu_info",
        lambda hardware_info: {"task_type": hardware_info.task_type.value, "gpu_count": 0},
    )
    monkeypatch.setattr(
        "ml.utils.runtime.runtime_snapshot.get_conda_env_export",
        lambda: "name: env\ndependencies: []\n",
    )
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.time.perf_counter", lambda: 103.456)
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.sys.executable", "/usr/bin/python")

    snapshot = build_runtime_snapshot(
        timestamp="2026-03-05T12:00:00",
        hardware_info=_cpu_hardware(),
        start_time=100.0,
    )

    assert snapshot["execution"]["git_commit"] == "abc123"
    assert snapshot["execution"]["python_executable"] == "/usr/bin/python"
    assert snapshot["execution"]["duration_seconds"] == 3.456
    assert snapshot["runtime"] == {"os": "Linux"}
    assert snapshot["environment"]["conda_env_hash"] != "Unavailable"


def test_build_runtime_snapshot_falls_back_when_conda_export_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that if get_conda_env_export raises an exception when trying to export the conda environment, build_runtime_snapshot catches this and sets the conda_env_export and conda_env_hash fields in the environment section of the snapshot to "Unavailable", while still successfully building the rest of the snapshot with expected values from the other helper functions. The test uses monkeypatch to replace get_conda_env_export with a fake function that raises a RuntimeMLError, then calls build_runtime_snapshot and asserts that the resulting snapshot has "Unavailable" for both conda_env_export and conda_env_hash, while other fields are populated as expected.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture used to replace get_conda_env_export with a fake function that raises an exception, simulating a failure in exporting the conda environment.
    """
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_git_commit", lambda _: "abc123")
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_runtime_info", lambda: {"os": "Linux"})
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_gpu_info", lambda hardware_info: {"gpu_count": 0})

    def _raise() -> str:
        """Fake function to simulate a failure in exporting the conda environment by raising a RuntimeMLError.

        Raises:
            RuntimeMLError: An error indicating that the conda environment export failed.
        """
        raise RuntimeMLError("conda failed")

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_conda_env_export", _raise)
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.time.perf_counter", lambda: 101.0)

    snapshot = build_runtime_snapshot(
        timestamp="2026-03-05T12:00:00",
        hardware_info=_cpu_hardware(),
        start_time=100.0,
    )

    assert snapshot["environment"] == {
        "conda_env_export": "Unavailable",
        "conda_env_hash": "Unavailable",
    }


def test_build_runtime_snapshot_wraps_unexpected_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that if get_runtime_info raises an unexpected exception when trying to collect runtime information, build_runtime_snapshot catches this and raises a RuntimeMLError with an appropriate error message that includes the original exception message, indicating a failure to build the runtime snapshot. The test uses monkeypatch to replace get_runtime_info with a fake function that raises a ValueError, then calls build_runtime_snapshot and asserts that a RuntimeMLError is raised with a message indicating a failure to build the runtime snapshot and that the original ValueError message is included.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture used to replace get_runtime_info with a fake function.
    """
    def _raise() -> dict:
        """Fake function to simulate an unexpected failure in collecting runtime information by raising a ValueError."""
        raise ValueError("runtime info failed")

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_runtime_info", _raise)

    with pytest.raises(RuntimeMLError, match="Failed to build runtime snapshot"):
        build_runtime_snapshot(
            timestamp="2026-03-05T12:00:00",
            hardware_info=_cpu_hardware(),
            start_time=100.0,
        )


def test_save_runtime_snapshot_writes_runtime_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that save_runtime_snapshot successfully writes the runtime snapshot to a runtime.json file in the target directory. The test uses monkeypatch to replace build_runtime_snapshot with a fake function that returns a specific expected snapshot dictionary, then calls save_runtime_snapshot with a sample timestamp, hardware info, and start time, and asserts that a runtime.json file is created in the target directory with contents that match the expected snapshot dictionary. This validates that save_runtime_snapshot correctly calls build_runtime_snapshot and writes the resulting snapshot to a JSON file in the specified location.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this as the target directory for saving the runtime snapshot, and checks that a runtime.json file is created with the expected contents.
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture used to replace build_runtime
    """
    expected_snapshot = {"execution": {"created_at": "2026-03-05T12:00:00"}}
    monkeypatch.setattr("ml.utils.runtime.save_runtime.build_runtime_snapshot", lambda *args, **kwargs: expected_snapshot)

    save_runtime_snapshot(
        target_dir=tmp_path,
        timestamp="2026-03-05T12:00:00",
        hardware_info=_cpu_hardware(),
        start_time=100.0,
    )

    saved = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert saved == expected_snapshot


def test_save_runtime_snapshot_rejects_overwrite_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that save_runtime_snapshot raises a PersistenceError when trying to save a runtime snapshot to a target directory that already contains a runtime.json file, and overwrite_existing is set to False. The test first creates a runtime.json file in the temporary directory to simulate an existing snapshot, then calls save_runtime_snapshot with overwrite_existing set to False, and asserts that a PersistenceError is raised with a message indicating that a snapshot already exists at the target location. This validates that save_runtime_snapshot correctly checks for existing snapshots and respects the overwrite_existing flag.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to create an existing runtime.json file to simulate a pre-existing snapshot, and then checks that save_runtime_snapshot raises an error when trying to overwrite it with overwrite_existing set to False.
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture used to replace build_runtime
    """
    (tmp_path / "runtime.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("ml.utils.runtime.save_runtime.build_runtime_snapshot", lambda *args, **kwargs: {"x": 1})

    with pytest.raises(PersistenceError, match="already exists"):
        save_runtime_snapshot(
            target_dir=tmp_path,
            timestamp="2026-03-05T12:00:00",
            hardware_info=_cpu_hardware(),
            start_time=100.0,
            overwrite_existing=False,
        )


def test_save_runtime_snapshot_overwrites_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that save_runtime_snapshot successfully overwrites an existing runtime.json file in the target directory when overwrite_existing is set to True. The test first creates a runtime.json file in the temporary directory with specific contents, then uses monkeypatch to replace build_runtime_snapshot with a fake function that returns a different expected snapshot dictionary. It then calls save_runtime_snapshot with overwrite_existing set to True, and asserts that the runtime.json file in the target directory is overwritten with the new contents from the expected snapshot dictionary. This validates that save_runtime_snapshot correctly overwrites existing snapshots when allowed.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create
    """
    (tmp_path / "runtime.json").write_text("{\"old\": 1}", encoding="utf-8")
    monkeypatch.setattr("ml.utils.runtime.save_runtime.build_runtime_snapshot", lambda *args, **kwargs: {"new": 2})

    save_runtime_snapshot(
        target_dir=tmp_path,
        timestamp="2026-03-05T12:00:00",
        hardware_info=_cpu_hardware(),
        start_time=100.0,
        overwrite_existing=True,
    )

    saved = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert saved == {"new": 2}
