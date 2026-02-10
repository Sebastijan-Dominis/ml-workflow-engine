import logging

from ml.utils.git import get_git_commit, is_descendant_commit

logger = logging.getLogger(__name__)

def validate_git_commits_match(runtime_info: dict) -> None:
    git_commit = get_git_commit()
    expected_commit = runtime_info.get("execution", {}).get("git_commit", "<unknown>")

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