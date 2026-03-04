"""Git utility helpers for reproducibility and ancestry checks."""

import logging
import subprocess
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

MergeTarget = Literal["training", "model", "ensemble"]

def get_git_commit(repo_dir: Path = Path(".")) -> str:
    """Return the current HEAD commit hash for the repository or `unknown`.

    Args:
        repo_dir: Directory inside the target git repository.

    Returns:
        str: HEAD commit hash, or ``"unknown"`` when unavailable.
    """

    try:
        # Find the top-level git directory
        top_level = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        # Get the HEAD commit hash
        commit_hash = subprocess.check_output(
            ["git", "-C", top_level, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        return commit_hash
    except subprocess.CalledProcessError:
        return "unknown"

def is_descendant_commit(commit_a: str, commit_b: str) -> bool:
    """Return whether `commit_a` descends from `commit_b` in git history.

    Args:
        commit_a: Candidate descendant commit hash.
        commit_b: Candidate ancestor commit hash.

    Returns:
        bool: ``True`` when `commit_b` is an ancestor of `commit_a`.
    """

    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit_b, commit_a],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False
