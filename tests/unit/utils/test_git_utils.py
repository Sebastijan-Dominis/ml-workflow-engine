"""Unit tests for git utility helpers."""

import subprocess
from pathlib import Path

import pytest
from ml.utils.git import get_git_commit, is_descendant_commit

pytestmark = pytest.mark.unit


def test_get_git_commit_returns_hash_when_git_calls_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_git_commit returns the expected commit hash when the subprocess calls to retrieve the git repository root and current commit hash succeed. The test uses monkeypatch to replace subprocess.check_output with a fake function that returns specific byte strings for the repository root and commit hash, then calls get_git_commit and asserts that the returned commit hash matches the expected value from the fake subprocess output. This validates that get_git_commit correctly interacts with subprocess to retrieve git information and processes the output to extract the commit hash."""
    outputs = iter([b"/repo\n", b"abcdef123456\n"])
    monkeypatch.setattr("ml.utils.git.subprocess.check_output", lambda *args, **kwargs: next(outputs))

    result = get_git_commit(Path("."))

    assert result == "abcdef123456"


def test_get_git_commit_returns_unknown_on_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_git_commit returns "unknown" when the subprocess call to retrieve git information raises a CalledProcessError. The test uses monkeypatch to replace subprocess.check_output with a fake function that raises a CalledProcessError, then calls get_git_commit and asserts that the returned value is "unknown". This validates that get_git_commit correctly handles errors from subprocess calls and returns a fallback value when git information cannot be retrieved."""
    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["git"])

    monkeypatch.setattr("ml.utils.git.subprocess.check_output", _raise)

    result = get_git_commit(Path("."))

    assert result == "unknown"


def test_is_descendant_commit_returns_true_when_merge_base_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that is_descendant_commit returns True when the subprocess call to retrieve the merge base succeeds. The test uses monkeypatch to replace subprocess.run with a fake function that does nothing, then calls is_descendant_commit and asserts that the returned value is True. This validates that is_descendant_commit correctly identifies descendant commits when the subprocess call succeeds."""
    monkeypatch.setattr("ml.utils.git.subprocess.run", lambda *args, **kwargs: None)

    assert is_descendant_commit("child", "parent") is True


def test_is_descendant_commit_returns_false_on_called_process_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that is_descendant_commit returns False when the subprocess call to retrieve the merge base raises a CalledProcessError. The test uses monkeypatch to replace subprocess.run with a fake function that raises a CalledProcessError, then calls is_descendant_commit and asserts that the returned value is False. This validates that is_descendant_commit correctly handles errors from subprocess calls and returns a fallback value when the merge base cannot be retrieved."""
    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["git", "merge-base"])

    monkeypatch.setattr("ml.utils.git.subprocess.run", _raise)

    assert is_descendant_commit("child", "parent") is False
