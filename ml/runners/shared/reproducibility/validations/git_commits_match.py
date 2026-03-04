"""Reproducibility checks for git commit ancestry and branch alignment."""

import logging

from ml.runners.shared.models.runtime_info import RuntimeInfo
from ml.utils.git import get_git_commit, is_descendant_commit

logger = logging.getLogger(__name__)

def validate_git_commits_match(runtime_info: RuntimeInfo) -> None:
    """Compare current git commit to expected commit and log reproducibility risk.

    Args:
        runtime_info: Runtime metadata dictionary containing expected git commit.

    Returns:
        None.

    Notes:
        This check is advisory by design: it logs reproducibility risks for
        descendant/divergent commits and does not raise to avoid blocking
        post-hoc reproducibility inspection workflows.

    Side Effects:
        Executes git introspection helpers and emits warning/debug logs.
    """

    git_commit = get_git_commit()
    expected_commit = runtime_info.execution.git_commit

    if expected_commit != git_commit:
        # descendant commit → warn
        if is_descendant_commit(git_commit, expected_commit):
            logger.warning(
                f"Current git commit {git_commit} is a descendant of expected {expected_commit}. "
                "Reproducibility may be affected."
            )
        # different branch → warn strongly
        elif not is_descendant_commit(expected_commit, git_commit):
            logger.warning(
                f"Current git commit {git_commit} is on a different branch than expected {expected_commit}. "
                "Reproducibility may be significantly affected."
            )
    else:
        # same commit → OK
        logger.debug(f"Git commit matches expected: {git_commit}")
