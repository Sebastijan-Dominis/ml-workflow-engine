import json
import logging
from pathlib import Path

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)


def save_metadata(
    metadata: dict, 
    *, 
    target_dir: Path,
    overwrite_existing: bool = False
) -> None:
    metadata_file = target_dir / "metadata.json"

    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    if metadata_file.exists():
        if not overwrite_existing:
            msg = f"Metadata file already exists at {metadata_file}, and overwriting is disabled. Set overwrite_existing=True to overwrite it."
            logger.error(msg)
            raise PersistenceError(msg)
        logger.critical(f"Metadata file already exists at {metadata_file}, and overwriting is enabled. It will be overwritten.")

    try:
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata successfully saved to {metadata_file}.")
    except Exception as e:
        msg = f"Failed to save metadata to {metadata_file}"
        logger.exception(msg)
        raise PersistenceError(msg) from e