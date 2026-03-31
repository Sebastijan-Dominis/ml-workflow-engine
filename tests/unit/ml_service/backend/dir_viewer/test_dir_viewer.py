"""Tests for the directory tree builder utility.

These tests use `tmp_path` and a small fake path object to ensure behaviour
is consistent across Windows and Linux.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ml_service.backend.dir_viewer.utils.build_tree import build_tree


def test_build_tree_nested(tmp_path: Path) -> None:
    """`build_tree` returns nested dictionaries for directories and ``None`` for files."""

    a = tmp_path / "a"
    a.mkdir()
    (a / "file1.txt").write_text("hello")
    b = a / "b"
    b.mkdir()
    (b / "file2.txt").write_text("hi")

    tree = cast(dict[str, Any], build_tree(tmp_path))

    assert "a" in tree
    assert tree["a"]["file1.txt"] is None
    assert "b" in tree["a"]
    assert tree["a"]["b"]["file2.txt"] is None


def test_build_tree_permission_error() -> None:
    """When iteration raises PermissionError, `build_tree` returns an error dict."""

    class FakePath:
        def iterdir(self) -> Any:
            raise PermissionError

    result = build_tree(cast(Path, FakePath()))
    assert result == {"error": "Permission denied"}
