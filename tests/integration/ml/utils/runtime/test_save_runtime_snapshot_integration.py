"""Integration tests for `ml.utils.runtime.save_runtime_snapshot`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ml.config.schemas.hardware_cfg import HardwareConfig, HardwareTaskType
from ml.exceptions import PersistenceError
from ml.utils.runtime import save_runtime as saver


def test_save_runtime_snapshot_writes_file(tmp_path: Path, monkeypatch: Any) -> None:
    snapshot = {"alpha": 1}
    monkeypatch.setattr(saver, "build_runtime_snapshot", lambda *a, **k: snapshot)

    target = tmp_path / "runs" / "r1"
    hw = HardwareConfig(task_type=HardwareTaskType.CPU, devices=[])

    saver.save_runtime_snapshot(target_dir=target, timestamp="ts", hardware_info=hw, start_time=0.0)

    out = target / "runtime.json"
    assert out.exists()
    with out.open("r", encoding="utf-8") as f:
        got = json.load(f)
    assert got == snapshot


def test_save_runtime_snapshot_raises_if_exists_and_no_overwrite(tmp_path: Path, monkeypatch: Any) -> None:
    snapshot = {"alpha": 1}
    monkeypatch.setattr(saver, "build_runtime_snapshot", lambda *a, **k: snapshot)

    target = tmp_path / "runs" / "r2"
    target.mkdir(parents=True)
    existing = target / "runtime.json"
    existing.write_text("{}")

    hw = HardwareConfig(task_type=HardwareTaskType.CPU, devices=[])
    with pytest.raises(PersistenceError):
        saver.save_runtime_snapshot(target_dir=target, timestamp="ts", hardware_info=hw, start_time=0.0, overwrite_existing=False)


def test_save_runtime_snapshot_overwrites_when_flag_true(tmp_path: Path, monkeypatch: Any) -> None:
    snapshot = {"alpha": 2}
    monkeypatch.setattr(saver, "build_runtime_snapshot", lambda *a, **k: snapshot)

    target = tmp_path / "runs" / "r3"
    target.mkdir(parents=True)
    existing = target / "runtime.json"
    existing.write_text("{\"alpha\": 1}")

    hw = HardwareConfig(task_type=HardwareTaskType.CPU, devices=[])
    saver.save_runtime_snapshot(target_dir=target, timestamp="ts", hardware_info=hw, start_time=0.0, overwrite_existing=True)

    with (target / "runtime.json").open("r", encoding="utf-8") as f:
        got = json.load(f)
    assert got == snapshot
