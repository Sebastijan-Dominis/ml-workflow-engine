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


def test_delete_failure_management_folder_deletes_empty_parent_chain_for_train_stage(
    tmp_path: Path,
) -> None:
    """Delete run folder and prune empty train/experiment/main directories for train stage."""
    main_dir = tmp_path / "failure_management"
    experiment_dir = main_dir / "exp_train"
    train_dir = experiment_dir / "train"
    run_dir = train_dir / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")

    delete_failure_management_folder(folder_path=run_dir, cleanup=True, stage="train")

    assert not run_dir.exists()
    assert not train_dir.exists()
    assert not experiment_dir.exists()
    assert not main_dir.exists()


def test_delete_failure_management_folder_keeps_non_empty_train_parents_for_train_stage(
    tmp_path: Path,
) -> None:
    """Keep parent directories when train stage still has sibling runs after cleanup."""
    main_dir = tmp_path / "failure_management"
    experiment_dir = main_dir / "exp_train"
    train_dir = experiment_dir / "train"
    run_dir = train_dir / "run_001"
    sibling_run_dir = train_dir / "run_002"
    run_dir.mkdir(parents=True)
    sibling_run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")
    (sibling_run_dir / "marker.txt").write_text("keep", encoding="utf-8")

    delete_failure_management_folder(folder_path=run_dir, cleanup=True, stage="train")

    assert not run_dir.exists()
    assert train_dir.exists()
    assert experiment_dir.exists()
    assert main_dir.exists()
    assert sibling_run_dir.exists()


def test_delete_failure_management_folder_keeps_non_empty_search_parent(tmp_path: Path) -> None:
    """Keep search parent directory when sibling experiment folders remain."""
    main_dir = tmp_path / "failure_management"
    target_experiment_dir = main_dir / "exp_a"
    sibling_experiment_dir = main_dir / "exp_b"
    target_experiment_dir.mkdir(parents=True)
    sibling_experiment_dir.mkdir(parents=True)
    (target_experiment_dir / "state.json").write_text("{}", encoding="utf-8")
    (sibling_experiment_dir / "marker.txt").write_text("keep", encoding="utf-8")

    delete_failure_management_folder(folder_path=target_experiment_dir, cleanup=True, stage="search")

    assert not target_experiment_dir.exists()
    assert main_dir.exists()
    assert sibling_experiment_dir.exists()


def test_delete_failure_management_folder_logs_skip_when_cleanup_disabled(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit informational skip message when cleanup is explicitly disabled."""
    folder = tmp_path / "exp_log"
    folder.mkdir()

    with caplog.at_level(
        "INFO", logger="ml.search.utils.failure_management.delete_failure_management_folder"
    ):
        delete_failure_management_folder(folder_path=folder, cleanup=False, stage="search")

    assert "Skipping cleanup of failure management folder for experiment exp_log." in caplog.text


def test_delete_failure_management_folder_logs_warning_for_unexpected_subdirs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit safety warning message when unknown subdirectories prevent deletion."""
    folder = tmp_path / "exp_warn"
    folder.mkdir()
    (folder / "unexpected").mkdir()

    with caplog.at_level(
        "WARNING", logger="ml.search.utils.failure_management.delete_failure_management_folder"
    ):
        delete_failure_management_folder(folder_path=folder, cleanup=True, stage="search")

    assert "contains unexpected subdirectories" in caplog.text
