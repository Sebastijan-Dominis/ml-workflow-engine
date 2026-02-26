import logging
from pathlib import Path
from ml.registry.hash_registry import hash_data

from ml.exceptions import UserError

logger = logging.getLogger(__name__)

def validate_data(*, data_path: Path, metadata: dict) -> str:
    expected_hash = metadata.get("data", {}).get("hash")
    if expected_hash is None:
        logger.warning("No data hash found in metadata. Skipping data integrity check.")
        return ""
    
    actual_hash = hash_data(data_path)

    if expected_hash != actual_hash:
        msg = f"Data hash mismatch. Expected: {expected_hash}, Actual: {actual_hash}"
        logger.error(msg)
        raise UserError(msg)
    
    return actual_hash