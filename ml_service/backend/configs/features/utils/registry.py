"""Feature registry utilities."""

from pathlib import Path

import yaml


def load_registry(path: Path) -> dict:
    print(f"Loading feature registry from {path}")
    if not path.exists():
        raise RuntimeError(f"Feature registry missing: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise RuntimeError("Feature registry YAML is empty or corrupted.")

    if not isinstance(data, dict):
        raise RuntimeError("Feature registry must be a dict.")

    return data


def registry_entry_exists(name: str, version: str, registry_path: Path) -> bool:
    """Check whether feature set already exists."""

    registry = load_registry(registry_path)

    if name not in registry:
        return False

    return version in registry[name]
