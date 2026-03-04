"""Hashing helpers for Arrow file metadata integrity fingerprints."""

import hashlib
import logging
from pathlib import Path

import pyarrow as pa
from ml.exceptions import RuntimeMLError
from ml.features.hashing.safe import safe

logger = logging.getLogger(__name__)

def hash_arrow_metadata(path: Path) -> str:
    """Compute a deterministic hash from Arrow schema and batch metadata.

    Args:
        path: Path to the Arrow file to fingerprint.

    Returns:
        Deterministic metadata hash for the Arrow file.
    """

    try:
        with pa.memory_map(path, 'r') as source:
            reader = pa.ipc.open_file(source)
            schema = reader.schema

            h = hashlib.sha256()
            for field in schema:
                h.update(field.name.encode())
                h.update(safe(field.type).encode())
                h.update(safe(field.nullable).encode())

            h.update(safe(reader.num_record_batches).encode())
            return h.hexdigest()
    except Exception as e:
        msg = f"Failed to hash Arrow metadata for file {path}. "
        logger.error(msg + f"Details: {str(e)}")
        raise RuntimeMLError(msg) from e
