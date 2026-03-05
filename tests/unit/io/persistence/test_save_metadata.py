"""Unit tests for metadata persistence helper."""

import json
from pathlib import Path

import pytest
from ml.exceptions import PersistenceError
from ml.io.persistence.save_metadata import save_metadata

pytestmark = pytest.mark.unit


def test_save_metadata_writes_metadata_json(tmp_path: Path) -> None:
    """Test that save_metadata writes the correct metadata to a JSON file."""
    target_dir = tmp_path / "meta"
    metadata = {"run_id": "abc", "score": 0.91}

    save_metadata(metadata, target_dir=target_dir)

    saved = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved == metadata


def test_save_metadata_rejects_existing_file_when_overwrite_disabled(tmp_path: Path) -> None:
    """Test that save_metadata raises a PersistenceError when the target metadata.json file already exists and overwrite_existing is set to False. The test creates a temporary directory and writes an initial metadata.json file to it, then calls save_metadata with overwrite_existing=False and asserts that a PersistenceError is raised with a message indicating that the file already exists, confirming that save_metadata correctly prevents overwriting existing metadata files when overwrite_existing is disabled."""
    target_dir = tmp_path / "meta"
    target_dir.mkdir(parents=True)
    (target_dir / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PersistenceError, match="already exists"):
        save_metadata({"new": 1}, target_dir=target_dir, overwrite_existing=False)


def test_save_metadata_overwrites_existing_file_when_enabled(tmp_path: Path) -> None:
    """Test that save_metadata overwrites an existing metadata.json file when overwrite_existing is set to True."""
    target_dir = tmp_path / "meta"
    target_dir.mkdir(parents=True)
    (target_dir / "metadata.json").write_text('{"old": 1}', encoding="utf-8")

    save_metadata({"new": 2}, target_dir=target_dir, overwrite_existing=True)

    saved = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
    assert saved == {"new": 2}
