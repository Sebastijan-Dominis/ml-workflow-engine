"""Streaming file hashing utilities for large artifact integrity checks."""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def hash_streaming(path: Path, chunk_size=1024 * 1024) -> str:
    """Compute SHA-256 hash incrementally by reading file chunks.

    Args:
        path: File path to hash.
        chunk_size: Byte size for incremental read chunks.

    Returns:
        SHA-256 hash for the file content.
    """

    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        msg = f"Error hashing file {path} using streaming method. "
        logger.error(msg + f"Details: {str(e)}")
        raise RuntimeError(msg) from e
