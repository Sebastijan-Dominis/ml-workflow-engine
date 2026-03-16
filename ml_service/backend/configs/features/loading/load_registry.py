"""Utility for loading the feature registry YAML."""

from pathlib import Path

import yaml


def load_feature_registry(registry_path: Path) -> dict:
    """Load feature registry YAML safely.

    Args:
        registry_path: Path to the feature registry YAML file.

    Returns:
        Parsed feature registry as a dictionary.
    """

    if registry_path.exists():
        with open(registry_path) as f:
            return yaml.safe_load(f) or {}

    return {}
