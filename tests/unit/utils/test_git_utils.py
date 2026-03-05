"""Unit tests for git utility helpers."""

import subprocess
from pathlib import Path

import pytest
from ml.utils.git import get_git_commit, is_descendant_commit

pytestmark = pytest.mark.unit


def test_get_git_commit_returns_hash_when_git_calls_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `get_git_commit` returns the current commit hash on success."""
    outputs = iter([b"/repo\n", b"abcdef123456\n"])
    monkeypatch.setattr("ml.utils.git.subprocess.check_output", lambda *args, **kwargs: next(outputs))

    result = get_git_commit(Path("."))

    assert result == "abcdef123456"


def test_get_git_commit_returns_unknown_on_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `get_git_commit` returns `"unknown"` on subprocess failure."""
    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["git"])

    monkeypatch.setattr("ml.utils.git.subprocess.check_output", _raise)

    result = get_git_commit(Path("."))

    assert result == "unknown"


def test_is_descendant_commit_returns_true_when_merge_base_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `is_descendant_commit` returns `True` when merge-base succeeds."""
    monkeypatch.setattr("ml.utils.git.subprocess.run", lambda *args, **kwargs: None)

    assert is_descendant_commit("child", "parent") is True


def test_is_descendant_commit_returns_false_on_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `is_descendant_commit` returns `False` on subprocess failure."""
    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["git", "merge-base"])

    monkeypatch.setattr("ml.utils.git.subprocess.run", _raise)

    assert is_descendant_commit("child", "parent") is False
