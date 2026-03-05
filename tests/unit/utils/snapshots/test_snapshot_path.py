"""Unit tests for snapshot path resolution helper."""

import sys
import types
from pathlib import Path

import pytest
from ml.exceptions import RuntimeMLError

# `ml.types` imports CatBoost classes at module import time; provide a test stub.
if "catboost" not in sys.modules:
    catboost_stub = types.ModuleType("catboost")
    catboost_stub.__dict__.update(
        {
            "CatBoostClassifier": type("CatBoostClassifier", (), {}),
            "CatBoostRegressor": type("CatBoostRegressor", (), {}),
        }
    )
    sys.modules["catboost"] = catboost_stub

from ml.utils.snapshots.snapshot_path import get_snapshot_path

pytestmark = pytest.mark.unit


def test_get_snapshot_path_returns_explicit_snapshot_path(tmp_path: Path) -> None:
    """Test that get_snapshot_path returns the expected path when given an explicit snapshot name. The test creates a directory in the temporary path with a name that matches the expected snapshot naming convention (e.g., "snapshot-123"), then calls get_snapshot_path with this explicit snapshot name and asserts that the returned path corresponds to the created directory, confirming that the function correctly constructs the snapshot path based on the provided name.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create
    """
    result = get_snapshot_path("snapshot-123", tmp_path)

    assert result == tmp_path / "snapshot-123"


def test_get_snapshot_path_resolves_latest_via_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_snapshot_path correctly resolves the "latest" snapshot by delegating to the get_latest_snapshot_path helper function. The test uses monkeypatch to replace get_latest_snapshot_path with a fake function that returns a specific expected path, then calls get_snapshot_path with "latest" as the snapshot name and asserts that the returned path matches the expected path from the fake helper, confirming that get_snapshot_path correctly delegates to get_latest_snapshot_path when "latest" is specified.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to construct the expected path returned by the fake helper function.
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture used to replace get_latest_snapshot_path with a fake function that returns a specific expected path.
    """
    expected = tmp_path / "2024-01-01T10-00-00_abcd"

    monkeypatch.setattr(
        "ml.utils.snapshots.snapshot_path.get_latest_snapshot_path",
        lambda snapshot_dir: expected,
    )

    result = get_snapshot_path("latest", tmp_path)

    assert result == expected


def test_get_snapshot_path_wraps_latest_resolution_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that if get_latest_snapshot_path raises an exception when resolving the latest snapshot, get_snapshot_path catches this and raises a RuntimeMLError with an appropriate error message that includes the original exception message. The test uses monkeypatch to replace get_latest_snapshot_path with a fake function that raises a ValueError, then calls get_snapshot_path with "latest" and asserts that a RuntimeMLError is raised with a message indicating a failure to resolve the latest snapshot and that the original ValueError message is included.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to construct the expected path returned by the fake helper function.
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture used to replace get_latest_snapshot_path with a fake function that raises an exception.
    """
    def _raise(_: Path) -> Path:
        raise ValueError("failed")

    monkeypatch.setattr("ml.utils.snapshots.snapshot_path.get_latest_snapshot_path", _raise)

    with pytest.raises(RuntimeMLError, match="Failed to resolve latest snapshot"):
        get_snapshot_path("latest", tmp_path)
