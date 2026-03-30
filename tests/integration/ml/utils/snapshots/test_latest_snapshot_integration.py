"""Integration tests for snapshot resolution helpers in `ml.utils.snapshots`."""

from __future__ import annotations

from pathlib import Path

import pytest
from ml.exceptions import RuntimeMLError
from ml.types import LatestSnapshot
from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path
from ml.utils.snapshots.snapshot_path import get_snapshot_path


def test_get_latest_snapshot_path_selects_most_recent(tmp_path: Path) -> None:
    base = tmp_path / "snapshots"
    base.mkdir()

    old = base / "2026-03-30T12-00-00_aaaaaaaa"
    new = base / "2026-03-31T12-00-00_bbbbbbbb"
    old.mkdir()
    new.mkdir()

    got = get_latest_snapshot_path(base)
    assert got == new


def test_get_latest_snapshot_tie_breaks_by_uuid(tmp_path: Path) -> None:
    base = tmp_path / "snapshots2"
    base.mkdir()

    # same timestamp, different UUIDs -> lexicographic tie-break
    a = base / "2026-04-01T00-00-00_aaaa1111"
    b = base / "2026-04-01T00-00-00_zzzz9999"
    a.mkdir()
    b.mkdir()

    got = get_latest_snapshot_path(base)
    assert got == b


def test_get_snapshot_path_raises_runtime_when_no_valid(tmp_path: Path) -> None:
    base = tmp_path / "empty"
    base.mkdir()

    with pytest.raises(RuntimeMLError):
        get_snapshot_path(LatestSnapshot.LATEST.value, base)
