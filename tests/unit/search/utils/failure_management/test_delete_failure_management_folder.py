"""Unit tests for safe failure-management directory cleanup helper."""

from __future__ import annotations

from pathlib import Path

import pytest
from ml.search.utils.failure_management.delete_failure_management_folder import (
    delete_failure_management_folder,
)

pytestmark = pytest.mark.unit


def test_delete_failure_management_folder_skips_when_cleanup_disabled(tmp_path: Path) -> None:
    """Leave directories untouched when cleanup flag is disabled."""
    folder = tmp_path / "exp_a"
    folder.mkdir()
    (folder / "marker.txt").write_text("keep", encoding="utf-8")

    delete_failure_management_folder(folder_path=folder, cleanup=False, stage="search")

    assert folder.exists()
    assert (folder / "marker.txt").exists()


def test_delete_failure_management_folder_deletes_expected_search_tree(tmp_path: Path) -> None:
    """Delete search folder and empty parent when contents are safe-to-delete only."""
    main_dir = tmp_path / "failure_management"
    experiment_dir = main_dir / "exp_a"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "state.json").write_text("{}", encoding="utf-8")

    learn_dir = experiment_dir / "learn"
    learn_dir.mkdir()
    (learn_dir / "file.txt").write_text("x", encoding="utf-8")

    delete_failure_management_folder(folder_path=experiment_dir, cleanup=True, stage="search")

    assert not experiment_dir.exists()
    assert not main_dir.exists()


def test_delete_failure_management_folder_skips_when_unexpected_subdir_exists(tmp_path: Path) -> None:
    """Skip deletion for safety when directory contains unknown subdirectories."""
    folder = tmp_path / "exp_b"
    folder.mkdir()
    (folder / "unexpected_subdir").mkdir()
    marker = folder / "state.json"
    marker.write_text("{}", encoding="utf-8")

    delete_failure_management_folder(folder_path=folder, cleanup=True, stage="search")

    assert folder.exists()
    assert marker.exists()


def test_delete_failure_management_folder_skips_when_nested_dir_found_in_allowed_subdir(tmp_path: Path) -> None:
    """Abort cleanup when nested directories are discovered inside allowed subdirs."""
    folder = tmp_path / "exp_c"
    folder.mkdir()
    allowed_subdir = folder / "tmp"
    allowed_subdir.mkdir()
    (allowed_subdir / "nested").mkdir()

    delete_failure_management_folder(folder_path=folder, cleanup=True, stage="search")

    assert folder.exists()
    assert allowed_subdir.exists()
