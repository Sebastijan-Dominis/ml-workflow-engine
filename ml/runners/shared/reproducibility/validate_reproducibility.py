"""Orchestration entrypoint for reproducibility validation checks."""

import logging
from pathlib import Path

from ml.runners.shared.reproducibility.validations.conda_envs_match import validate_conda_envs_match
from ml.runners.shared.reproducibility.validations.git_commits_match import (
    validate_git_commits_match,
)
from ml.runners.shared.reproducibility.validations.runtime_comparison import (
    validate_runtime_matches,
)
from ml.runners.shared.reproducibility.validations.validate_runtime_info import (
    validate_runtime_info,
)
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def validate_reproducibility(runtime_info_path: Path) -> None:
    """Load runtime metadata and run git, environment, and runtime checks.

    Args:
        runtime_info_path: Path to runtime metadata JSON file.

    Returns:
        None.
    """

    runtime_info_raw = load_json(runtime_info_path)
    runtime_info = validate_runtime_info(runtime_info_raw)
    validate_git_commits_match(runtime_info)
    validate_conda_envs_match(runtime_info)
    validate_runtime_matches(runtime_info)
