"""A module for constructing file paths for pipeline configuration files based on data type, algorithm, and pipeline version."""

from pathlib import Path


def get_config_path(
        *,
        repo_root: str,
        data_type: str,
        algorithm: str,
        pipeline_version: str
    ) -> Path:
    """Construct the file path for a given config type, algorithm, and version.

    Args:
        repo_root (str): The root directory of the repository.
        data_type (str): The type of data (e.g., "tabular", "image").
        algorithm (str): The name of the algorithm (e.g., "random_forest").
        pipeline_version (str): The version of the pipeline (e.g., "v1.0").

    Returns:
        Path: The file path for the specified config.
    """
    return Path(repo_root) / "configs" / "pipelines" / data_type / algorithm / f"{pipeline_version}.yaml"
