import json
import logging
logger = logging.getLogger(__name__)

def save_metadata(path, metadata: dict):
    metadata_path = path / "metadata.json"
    json.dump(metadata, open(metadata_path, "w"), indent=4)
    logger.info(f"Saved metadata to {metadata_path}")