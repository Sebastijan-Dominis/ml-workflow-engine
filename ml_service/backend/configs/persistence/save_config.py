"""A module for saving data configuration files."""

import os
from pathlib import Path

import yaml
from fastapi import HTTPException


def save_config(config: dict, config_path: Path) -> None:
    """Save the config dict to the specified path, ensuring atomic write and handling errors.

    Args:
        config (dict): The configuration dictionary to save.
        config_path (Path): The file path where the config should be saved.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = config_path.parent / f"{config_path.name}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, config_path)

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Failed to write config: {str(e)}"
        ) from None
