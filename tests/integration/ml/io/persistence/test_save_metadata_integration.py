import json
from pathlib import Path

import pytest
from ml.exceptions import PersistenceError
from ml.io.persistence.save_metadata import save_metadata


def test_save_metadata_writes_file(tmp_path: Path) -> None:
    metadata = {"a": 1, "b": "x"}
    target_dir = tmp_path / "meta_dir"

    save_metadata(metadata, target_dir=target_dir, overwrite_existing=False)

    metadata_file = target_dir / "metadata.json"
    assert metadata_file.exists()

    with metadata_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data == metadata


def test_save_metadata_raises_on_existing_file_and_no_overwrite(tmp_path: Path) -> None:
    meta_dir = tmp_path / "meta_dir"
    meta_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = meta_dir / "metadata.json"
    metadata_file.write_text('{"existing": true}', encoding="utf-8")

    with pytest.raises(PersistenceError):
        save_metadata({"new": "value"}, target_dir=meta_dir, overwrite_existing=False)
