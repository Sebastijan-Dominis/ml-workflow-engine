from pathlib import Path
from typing import Any, cast

from ml_service.backend.dir_viewer.utils.build_tree import build_tree


def test_build_tree_returns_nested_structure(tmp_path: Path) -> None:
    base = tmp_path / "root"
    (base / "a").mkdir(parents=True)
    (base / "a" / "b").mkdir(parents=True)
    (base / "a" / "b" / "file.txt").write_text("x")
    (base / "other.txt").write_text("y")

    tree = cast(dict[str, Any], build_tree(base))

    assert "a" in tree
    assert "other.txt" in tree
    assert tree["a"]["b"]["file.txt"] is None
