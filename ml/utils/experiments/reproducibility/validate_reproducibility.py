"""Orchestration entrypoint for reproducibility validation checks."""

import logging
from pathlib import Path

from ml.utils.experiments.reproducibility.validations.conda_envs_match import validate_conda_envs_match
from ml.utils.experiments.reproducibility.validations.git_commits_match import validate_git_commits_match
from ml.utils.experiments.reproducibility.validations.runtime_comparison import validate_runtime
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def validate_reproducibility(runtime_info_path: Path) -> None:
    """Load runtime metadata and run git, environment, and runtime checks.

    Args:
        runtime_info_path: Path to runtime metadata JSON file.

    Returns:
        None.
    """

    runtime_info = load_json(runtime_info_path)
    validate_git_commits_match(runtime_info)
    validate_conda_envs_match(runtime_info)
    validate_runtime(runtime_info)