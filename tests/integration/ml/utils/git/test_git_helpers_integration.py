"""Integration tests for git helpers in `ml.utils.git`."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ml.utils.git import get_git_commit, is_descendant_commit


def test_get_git_commit_unknown(monkeypatch: Any) -> None:
    """When git calls fail, get_git_commit returns 'unknown'."""

    def fake_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd="git")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert get_git_commit(Path(".")) == "unknown"

def test_is_descendant_commit_true_and_false(monkeypatch: Any) -> None:
    """is_descendant_commit returns True on successful git run, False on error."""

    def fake_run_ok(*args, **kwargs):
        return None

    monkeypatch.setattr(subprocess, "run", fake_run_ok)
    assert is_descendant_commit("a", "b") is True

    def fake_run_fail(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd="git")

    monkeypatch.setattr(subprocess, "run", fake_run_fail)
    assert is_descendant_commit("a", "b") is False

def test_get_git_commit_success(monkeypatch: Any) -> None:
    """When git commands succeed, `get_git_commit` returns the HEAD hash."""

    def fake_check_output(*args, **kwargs):
        cmd = args[0]
        # git calls pass a list of command parts; inspect for markers
        if "--show-toplevel" in cmd:
            return b"/fake/top\n"
        if "HEAD" in cmd:
            return b"deadbeef\n"
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert get_git_commit(Path(".")) == "deadbeef"
