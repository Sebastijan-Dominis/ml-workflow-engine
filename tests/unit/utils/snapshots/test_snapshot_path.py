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
    """Verify that an explicit snapshot name resolves to the expected path."""
    result = get_snapshot_path("snapshot-123", tmp_path)

    assert result == tmp_path / "snapshot-123"


def test_get_snapshot_path_resolves_latest_via_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `latest` resolution delegates to the latest-snapshot helper."""
    expected = tmp_path / "2024-01-01T10-00-00_abcd"

    monkeypatch.setattr(
        "ml.utils.snapshots.snapshot_path.get_latest_snapshot_path",
        lambda snapshot_dir: expected,
    )

    result = get_snapshot_path("latest", tmp_path)

    assert result == expected


def test_get_snapshot_path_wraps_latest_resolution_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that latest-snapshot resolution errors are wrapped as `RuntimeMLError`."""
    def _raise(_: Path) -> Path:
        raise ValueError("failed")

    monkeypatch.setattr("ml.utils.snapshots.snapshot_path.get_latest_snapshot_path", _raise)

    with pytest.raises(RuntimeMLError, match="Failed to resolve latest snapshot"):
        get_snapshot_path("latest", tmp_path)
