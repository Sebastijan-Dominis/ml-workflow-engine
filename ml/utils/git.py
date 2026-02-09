import logging
import subprocess
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

MergeTarget = Literal["training", "model", "ensemble"]

def get_git_commit(repo_dir: Path = Path(".")) -> str:
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
    """Return True if commit_a is a descendant of commit_b."""
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