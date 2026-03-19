"""A module for checking if promotion thresholds already exist for a given problem type and segment."""
from pathlib import Path

import yaml


def check_thresholds_exist(
    config_path: Path,
    problem_type: str,
    segment: str
) -> tuple[bool, dict]:
    """Check if thresholds already exist for the given problem type and segment.

    Args:
        config_path (Path): The path to the thresholds configuration file.
        problem_type (str): The problem type to check.
        segment (str): The segment to check.

    Returns:
        tuple[bool, dict]: A tuple where the first element indicates if the thresholds exist,
                           and the second element is the full thresholds dictionary.
    """
    if not config_path.exists():
        return False, {}

    with open(config_path) as f:
        thresholds = yaml.safe_load(f) or {}

    target_thresholds = thresholds.get(problem_type, {}).get(segment)

    return target_thresholds is not None, thresholds
