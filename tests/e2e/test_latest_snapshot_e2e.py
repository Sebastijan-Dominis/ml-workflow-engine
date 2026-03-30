"""End-to-end tests for snapshot discovery utilities.

These tests create a realistic snapshot layout on disk and validate that the
`get_latest_snapshot_path` utility chooses the newest snapshot and handles
tie-breaking by UUID. They also assert that an empty directory raises the
expected ``DataError``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ml.exceptions import DataError
from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path


def _mk_snapshot(base: Path, ts: str, uuid_str: str) -> Path:
    p = base / f"{ts}_{uuid_str}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def test_get_latest_snapshot_with_tie_breaking(tmp_path: Path) -> None:
    """Latest snapshot selection prefers newest timestamp then highest UUID."""

    base = tmp_path / "snapshots"
    base.mkdir()

    ts1 = "2026-03-28T12-00-00"
    _mk_snapshot(base, ts1, "aaaaaaaa")
    p2 = _mk_snapshot(base, ts1, "bbbbbbbb")

    ts2 = "2026-03-28T13-00-00"
    p3 = _mk_snapshot(base, ts2, "cccccccc")

    selected = get_latest_snapshot_path(base)
    assert selected.name == p3.name

    # remove the newest and ensure the tie-breaking among the remaining works
    p3.rmdir()
    selected2 = get_latest_snapshot_path(base)
    assert selected2.name == p2.name


def test_get_latest_snapshot_raises_on_empty(tmp_path: Path) -> None:
    """An empty snapshots directory should raise ``DataError``."""

    base = tmp_path / "empty_snapshots"
    base.mkdir()

    with pytest.raises(DataError):
        get_latest_snapshot_path(base)
