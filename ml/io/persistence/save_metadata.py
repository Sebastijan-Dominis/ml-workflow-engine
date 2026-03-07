"""Persistence utilities for writing experiment metadata artifacts."""

import json
import logging
import os
import tempfile
from pathlib import Path

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)


def save_metadata(
    metadata: dict,
    *,
    target_dir: Path,
    overwrite_existing: bool = False
) -> None:
    """Persist metadata JSON to target directory with overwrite safeguards.

    Args:
        metadata: Metadata dictionary to persist.
        target_dir: Target directory for ``metadata.json``.
        overwrite_existing: Whether existing metadata file may be overwritten.

    Returns:
        None.

    Raises:
        PersistenceError: If overwrite policy is violated or file serialization
            fails.

    Side Effects:
        Creates target directories as needed and writes/overwrites
        ``metadata.json`` on disk.

    Examples:
        >>> save_metadata(metadata, target_dir=Path("/path/to/metadata"), overwrite_existing=False)
    """

    metadata_file = target_dir / "metadata.json"

    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    if metadata_file.exists():
        if not overwrite_existing:
            msg = f"Metadata file already exists at {metadata_file}, and overwriting is disabled. Set overwrite_existing=True to overwrite it."
            logger.error(msg)
            raise PersistenceError(msg)
        logger.critical(f"Metadata file already exists at {metadata_file}, and overwriting is enabled. It will be overwritten.")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=metadata_file.parent,
            prefix="metadata.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            temp_path = tmp_file.name
            json.dump(metadata, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(temp_path, metadata_file)
        logger.info(f"Metadata successfully saved to {metadata_file}.")
    except Exception as e:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except OSError:
                logger.warning("Failed to clean up temporary metadata file: %s", temp_path)

        msg = f"Failed to save metadata to {metadata_file}"
        logger.exception(msg)
        raise PersistenceError(msg) from e
