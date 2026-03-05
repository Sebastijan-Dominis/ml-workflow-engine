"""Unit tests for latest snapshot discovery logic."""

from pathlib import Path

import pytest
from ml.exceptions import DataError
from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path

pytestmark = pytest.mark.unit


def test_get_latest_snapshot_path_selects_newest_timestamp(tmp_path: Path) -> None:
    """Test that get_latest_snapshot_path correctly identifies the snapshot with the newest timestamp as the latest snapshot. The test creates multiple directories in the temporary path with names that follow the expected snapshot naming convention (timestamp_uuid), ensuring that at least two snapshots have different timestamps. Then it calls get_latest_snapshot_path with the temporary path and asserts that the returned path corresponds to the snapshot with the most recent timestamp, confirming that the function correctly parses timestamps and identifies the latest one.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create snapshot directories in. The test uses this to create multiple snapshot directories with different timestamps for testing.
    """
    (tmp_path / "2024-01-01T10-00-00_aaaa").mkdir()
    newest = tmp_path / "2024-01-02T09-00-00_bbbb"
    newest.mkdir()

    result = get_latest_snapshot_path(tmp_path)

    assert result == newest


def test_get_latest_snapshot_path_breaks_ties_by_uuid(tmp_path: Path) -> None:
    """Test that get_latest_snapshot_path breaks ties between snapshots with the same timestamp by selecting the one with the lexicographically highest UUID. The test creates multiple directories in the temporary path with names that follow the expected snapshot naming convention (timestamp_uuid), ensuring that at least two snapshots have the same timestamp but different UUIDs. Then it calls get_latest_snapshot_path with the temporary path and asserts that the returned path corresponds to the snapshot with the lexicographically highest UUID among those with the same timestamp, confirming that the function correctly implements tie-breaking logic based on UUID.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create snapshot directories in. The test uses this to create multiple snapshot directories with the same timestamp but different UUIDs for testing.
    """
    lower = tmp_path / "2024-01-03T11-00-00_0001"
    higher = tmp_path / "2024-01-03T11-00-00_ffff"
    lower.mkdir()
    higher.mkdir()

    result = get_latest_snapshot_path(tmp_path)

    assert result == higher


def test_get_latest_snapshot_path_ignores_invalid_entries(tmp_path: Path) -> None:
    """Test that get_latest_snapshot_path ignores entries in the snapshot directory that do not follow the expected snapshot naming convention (timestamp_uuid) and still correctly identifies the latest snapshot among the valid entries. The test creates several directories in the temporary path, including some with valid snapshot names and others with invalid names (e.g., missing timestamp, missing UUID, or completely unrelated names). Then it calls get_latest_snapshot_path with the temporary path and asserts that the returned path corresponds to the latest valid snapshot, confirming that the function correctly filters out invalid entries and does not consider them when determining the latest snapshot.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create snapshot directories in. The test uses this to create both valid and invalid entries for testing.
    """
    (tmp_path / "README.txt").write_text("not a dir", encoding="utf-8")
    (tmp_path / "invalidfolder").mkdir()
    valid = tmp_path / "2024-01-01T12-30-00_abcd"
    valid.mkdir()

    result = get_latest_snapshot_path(tmp_path)

    assert result == valid


def test_get_latest_snapshot_path_raises_when_no_valid_snapshots(tmp_path: Path) -> None:
    """Test that get_latest_snapshot_path raises a DataError when there are no valid snapshot directories in the specified path. The test creates several entries in the temporary path that do not follow the expected snapshot naming convention (e.g., files, directories with invalid names) and ensures that there are no valid snapshot directories. Then it calls get_latest_snapshot_path with the temporary path and asserts that a DataError is raised with an appropriate error message indicating that no valid snapshots were found, confirming that the function correctly handles the case where there are no valid snapshots to select from.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create invalid entries in. The test uses this to create entries that do not follow the snapshot naming convention, ensuring that there are no valid snapshots for testing.
    """
    (tmp_path / "bad").mkdir()

    with pytest.raises(DataError, match="No valid snapshots found"):
        get_latest_snapshot_path(tmp_path)
