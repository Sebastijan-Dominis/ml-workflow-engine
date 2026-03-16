"""Path utilities for feature registry."""

from pathlib import Path


def get_registry_path(repo_root: Path) -> Path:
    """Return feature registry YAML path."""

    return repo_root / "configs" / "feature_registry" / "features.yaml"
