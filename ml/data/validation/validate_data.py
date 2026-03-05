"""Data integrity validation helpers based on persisted hashes."""

import logging
from pathlib import Path

from ml.exceptions import UserError
from ml.utils.hashing.service import hash_data

logger = logging.getLogger(__name__)

def validate_data(*, data_path: Path, metadata: dict) -> str:
    """Validate dataset hash against metadata and return the computed hash.

    Args:
        data_path: Dataset file path.
        metadata: Metadata dictionary containing expected data hash.

    Returns:
        Computed data hash, or empty string when no expected hash is present.
    """

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
