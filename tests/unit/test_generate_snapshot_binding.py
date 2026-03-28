"""Unit tests for scripts.generators.generate_snapshot_binding.

These tests cover scanning for snapshots, loading/saving bindings,
and the module `main()` success/failure paths. Tests are written
to be cross-platform (Windows/Linux) and use pytest fixtures.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from scripts.generators import generate_snapshot_binding as gsb


def test_scan_latest_snapshots_nonexistent(tmp_path: Path) -> None:
    """scan_latest_snapshots returns empty dict when base dir missing."""
    base = tmp_path / "nonexistent"
    assert not base.exists()
    assert gsb.scan_latest_snapshots(base) == {}


def test_scan_latest_snapshots_finds_snapshots(tmp_path: Path, monkeypatch) -> None:
    """scan_latest_snapshots returns snapshot names using helper function."""
    base = tmp_path / "data"
    (base / "dataset1" / "v1").mkdir(parents=True)
    monkeypatch.setattr(gsb, "get_latest_snapshot_path", lambda _: Path("snap-v1"))
    res = gsb.scan_latest_snapshots(base)
    assert res == {"dataset1": {"v1": "snap-v1"}}


def test_scan_latest_snapshots_handles_data_error(tmp_path: Path, monkeypatch) -> None:
    """When get_latest_snapshot_path raises DataError, the version is skipped."""
    base = tmp_path / "data_err"
    (base / "datasetX" / "vX").mkdir(parents=True)

    def _raise(_: Path) -> Any:
        raise gsb.DataError("no snapshots")

    monkeypatch.setattr(gsb, "get_latest_snapshot_path", _raise)
    res = gsb.scan_latest_snapshots(base)
    assert res == {"datasetX": {}}


def test_load_bindings_empty_and_reads(tmp_path: Path) -> None:
    """load_bindings returns {} for missing file and parses YAML when present."""
    p = tmp_path / "bindings.yaml"
    assert gsb.load_bindings(p) == {}
    data = {"a": 1, "b": {"c": 2}}
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert gsb.load_bindings(p) == data


def test_save_bindings_atomic_success(tmp_path: Path) -> None:
    """save_bindings_atomic writes YAML atomically and the final file contains data."""
    p = tmp_path / "configs" / "snapshot_bindings_registry" / "bindings.yaml"
    data = {"k": "v"}
    gsb.save_bindings_atomic(p, data)
    assert p.exists()
    assert yaml.safe_load(p.read_text(encoding="utf-8")) == data


def test_save_bindings_atomic_failure_cleans_tmp(tmp_path: Path, monkeypatch) -> None:
    """If writing fails, the temporary file is removed and RuntimeMLError is raised."""
    p = tmp_path / "configs" / "snapshot_bindings_registry" / "bindings.yaml"
    data = {"k": "v"}

    def _bad_replace(_: Path, __: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", _bad_replace)
    with pytest.raises(gsb.RuntimeMLError):
        gsb.save_bindings_atomic(p, data)
    tmp_file = p.parent / f"{p.name}.tmp"
    assert not tmp_file.exists()
    assert not p.exists()


def test_main_success_writes_bindings(tmp_path: Path, monkeypatch) -> None:
    """main writes a new binding entry and returns 0 on success."""
    monkeypatch.setattr(gsb, "DATA_PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(gsb, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    bindings_path = tmp_path / "configs" / "snapshot_bindings_registry" / "bindings.yaml"
    monkeypatch.setattr(gsb, "BINDINGS_PATH", bindings_path)

    (gsb.DATA_PROCESSED_DIR / "dataset1" / "v1").mkdir(parents=True)
    (gsb.FEATURE_STORE_DIR / "featureA" / "v1").mkdir(parents=True)

    monkeypatch.setattr(gsb, "get_latest_snapshot_path", lambda _: Path("snap"))
    monkeypatch.setattr(gsb, "iso_no_colon", lambda _: "20260328T000000")
    monkeypatch.setattr(gsb, "uuid4", lambda: SimpleNamespace(hex="deadbeefcafebabe"))
    monkeypatch.setattr(gsb, "setup_logging", lambda *_, **__: None)

    rc = gsb.main()
    assert rc == 0

    doc = yaml.safe_load(bindings_path.read_text(encoding="utf-8"))
    expected_run_id = "20260328T000000_deadbeef"
    assert expected_run_id in doc
    entry = doc[expected_run_id]
    assert "datasets" in entry and "feature_sets" in entry
    assert entry["datasets"]["dataset1"]["v1"]["snapshot"] == "snap"
    assert entry["feature_sets"]["featureA"]["v1"]["snapshot"] == "snap"


def test_main_failure_returns_nonzero(tmp_path: Path, monkeypatch) -> None:
    """main returns non-zero when saving bindings fails."""
    monkeypatch.setattr(gsb, "DATA_PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(gsb, "FEATURE_STORE_DIR", tmp_path / "feature_store")
    monkeypatch.setattr(
        gsb,
        "BINDINGS_PATH",
        tmp_path / "configs" / "snapshot_bindings_registry" / "bindings.yaml",
    )

    (gsb.DATA_PROCESSED_DIR / "d1" / "v1").mkdir(parents=True)
    (gsb.FEATURE_STORE_DIR / "f1" / "v1").mkdir(parents=True)

    monkeypatch.setattr(gsb, "get_latest_snapshot_path", lambda _: Path("snap"))
    monkeypatch.setattr(gsb, "iso_no_colon", lambda _: "20260328T000000")
    monkeypatch.setattr(gsb, "uuid4", lambda: SimpleNamespace(hex="aaaaaaaa11111111"))
    monkeypatch.setattr(gsb, "setup_logging", lambda *_, **__: None)

    def _raise(_: Path, __: dict) -> None:
        raise gsb.RuntimeMLError("boom")

    monkeypatch.setattr(gsb, "save_bindings_atomic", _raise)
    rc = gsb.main()
    assert rc == 1
