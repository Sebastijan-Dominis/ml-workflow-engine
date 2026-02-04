import logging
logger = logging.getLogger(__name__)

import subprocess
from pathlib import Path
from typing import Literal

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
