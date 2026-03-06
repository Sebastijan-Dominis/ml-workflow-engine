"""Unit tests for latest snapshot discovery logic."""

from pathlib import Path

import pytest
from ml.exceptions import DataError
from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path

pytestmark = pytest.mark.unit


def test_get_latest_snapshot_path_selects_newest_timestamp(tmp_path: Path) -> None:
    """Verify that the newest timestamp is selected as the latest snapshot."""
    (tmp_path / "2024-01-01T10-00-00_aaaa").mkdir()
    newest = tmp_path / "2024-01-02T09-00-00_bbbb"
    newest.mkdir()

    result = get_latest_snapshot_path(tmp_path)

    assert result == newest


def test_get_latest_snapshot_path_breaks_ties_by_uuid(tmp_path: Path) -> None:
    """Verify that timestamp ties are resolved using lexicographic UUID ordering."""
    lower = tmp_path / "2024-01-03T11-00-00_0001"
    higher = tmp_path / "2024-01-03T11-00-00_ffff"
    lower.mkdir()
    higher.mkdir()

    result = get_latest_snapshot_path(tmp_path)

    assert result == higher


def test_get_latest_snapshot_path_ignores_invalid_entries(tmp_path: Path) -> None:
    """Verify that invalid directory entries are ignored during latest-snapshot selection."""
    (tmp_path / "README.txt").write_text("not a dir", encoding="utf-8")
    (tmp_path / "invalidfolder").mkdir()
    valid = tmp_path / "2024-01-01T12-30-00_abcd"
    valid.mkdir()

    result = get_latest_snapshot_path(tmp_path)

    assert result == valid


def test_get_latest_snapshot_path_raises_when_no_valid_snapshots(tmp_path: Path) -> None:
    """Verify that a `DataError` is raised when no valid snapshots are present."""
    (tmp_path / "bad").mkdir()

    with pytest.raises(DataError, match="No valid snapshots found"):
        get_latest_snapshot_path(tmp_path)


def test_get_latest_snapshot_path_ignores_timestamp_without_t_separator(tmp_path: Path) -> None:
    """Ignore folders with underscore split but invalid timestamp format lacking `T`."""
    (tmp_path / "2024-01-01-12-30-00_abcd").mkdir()
    valid = tmp_path / "2024-01-01T12-30-00_ffff"
    valid.mkdir()

    result = get_latest_snapshot_path(tmp_path)

    assert result == valid
