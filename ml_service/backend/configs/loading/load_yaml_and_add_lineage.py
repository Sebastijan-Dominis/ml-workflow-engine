"""A module for loading YAML configs and adding lineage information."""

import yaml

from ml_service.backend.configs.formatting.timestamp import add_timestamp


def load_yaml_and_add_lineage(yaml_text: str) -> dict:
    """Parse YAML and inject timestamp.

    Args:
        yaml_text: YAML string payload

    Returns:
        dict: YAML parsed into dict with lineage.created_at
    """

    data = yaml.safe_load(yaml_text)

    return add_timestamp(data, "lineage")
