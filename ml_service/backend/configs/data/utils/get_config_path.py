"""A module for getting the path to a config file based on the repository structure and input parameters."""
from pathlib import Path


def get_config_path(
        *,
        repo_root: str, 
        config_type: str, 
        dataset_name: str, 
        dataset_version: str
    ) -> Path:
    """Return path for a given config type, dataset, and version.

    Args:
        repo_root: Root directory of the repository
        config_type: "interim" or "processed"
        dataset_name: Name of the dataset
        dataset_version: Version of the dataset

    Returns:
        Path object pointing to the config file location
    """
    return Path(repo_root) / "configs" / "data" / config_type / dataset_name / f"{dataset_version}.yaml"