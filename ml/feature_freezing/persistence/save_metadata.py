import json
import logging
logger = logging.getLogger(__name__)
from pathlib import Path

from ml.exceptions import PersistenceError

def save_metadata(path: Path, metadata: dict):
    metadata_path = path / "metadata.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved metadata to {metadata_path}")
    except Exception as e:
        logger.exception("Failed to save metadata")
        raise PersistenceError(f"Could not save metadata to {metadata_path}") from e
