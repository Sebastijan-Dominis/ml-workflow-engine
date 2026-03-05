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
    """Build a CPU-only `HardwareConfig` for tests."""
    return HardwareConfig.model_validate({"task_type": "CPU", "devices": []})


def test_hash_environment_is_deterministic() -> None:
    """Verify that `hash_environment` is deterministic and SHA-256 sized."""
    env_export = "name: test-env\ndependencies:\n  - python=3.11\n"

    digest_a = hash_environment(env_export)
    digest_b = hash_environment(env_export)

    assert digest_a == digest_b
    assert len(digest_a) == 64


def test_build_runtime_snapshot_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify successful snapshot assembly when helpers return valid data."""
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
    """Verify fallback environment fields when conda export is unavailable."""
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_git_commit", lambda _: "abc123")
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_runtime_info", lambda: {"os": "Linux"})
    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_gpu_info", lambda hardware_info: {"gpu_count": 0})

    def _raise() -> str:
        """Simulate conda export failure."""
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
    """Verify that unexpected runtime collection errors are wrapped."""
    def _raise() -> dict:
        """Simulate runtime-info collection failure."""
        raise ValueError("runtime info failed")

    monkeypatch.setattr("ml.utils.runtime.runtime_snapshot.get_runtime_info", _raise)

    with pytest.raises(RuntimeMLError, match="Failed to build runtime snapshot"):
        build_runtime_snapshot(
            timestamp="2026-03-05T12:00:00",
            hardware_info=_cpu_hardware(),
            start_time=100.0,
        )


def test_save_runtime_snapshot_writes_runtime_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `save_runtime_snapshot` writes `runtime.json` to the target directory."""
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
    """Verify overwrite protection when `runtime.json` already exists."""
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
    """Verify overwrite behavior when `overwrite_existing=True`."""
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
