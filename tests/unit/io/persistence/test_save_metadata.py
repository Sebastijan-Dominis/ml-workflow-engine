"""Unit tests for metadata persistence helper."""

import json
from pathlib import Path

import pytest
from ml.exceptions import PersistenceError
from ml.io.persistence.save_metadata import save_metadata

pytestmark = pytest.mark.unit


def test_save_metadata_writes_metadata_json(tmp_path: Path) -> None:
    """Write metadata payload to `metadata.json` in the target directory."""
    target_dir = tmp_path / "meta"
    metadata = {"run_id": "abc", "score": 0.91}

    save_metadata(metadata, target_dir=target_dir)

    saved = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved == metadata


def test_save_metadata_rejects_existing_file_when_overwrite_disabled(tmp_path: Path) -> None:
    """Raise `PersistenceError` when target file exists and overwrite is disabled."""
    target_dir = tmp_path / "meta"
    target_dir.mkdir(parents=True)
    (target_dir / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PersistenceError, match="already exists"):
        save_metadata({"new": 1}, target_dir=target_dir, overwrite_existing=False)


def test_save_metadata_overwrites_existing_file_when_enabled(tmp_path: Path) -> None:
    """Overwrite existing `metadata.json` when overwrite is enabled."""
    target_dir = tmp_path / "meta"
    target_dir.mkdir(parents=True)
    (target_dir / "metadata.json").write_text('{"old": 1}', encoding="utf-8")

    save_metadata({"new": 2}, target_dir=target_dir, overwrite_existing=True)

    saved = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved == {"new": 2}
