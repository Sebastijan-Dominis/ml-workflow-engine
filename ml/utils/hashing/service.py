"""Hash function registries and convenience wrappers for artifacts/data.

The wrappers ``hash_file``, ``hash_data``, and ``hash_artifact`` are
intentional semantic aliases over the same streaming hash implementation.
They keep call sites domain-explicit while preserving a single hashing
mechanism.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from ml.exceptions import ConfigError
from ml.features.hashing.hash_arrow_metadata import hash_arrow_metadata
from ml.features.hashing.hash_parquet_metadata import hash_parquet_metadata
from ml.utils.hashing.hash_dict import hash_dict
from ml.utils.hashing.hash_streaming import hash_streaming

logger = logging.getLogger(__name__)

HASH_LOADER_REGISTRY: dict[str, Callable[[Path], str]] = {
    "parquet": hash_parquet_metadata,
    "arrow": hash_arrow_metadata,
    "csv": hash_streaming,
    "json": hash_streaming,
}

def hash_file(file_path: Path) -> str:
    """Compute SHA-256 hash of a file using streaming reads.

    Args:
        file_path: File path to hash.

    Returns:
        str: SHA-256 hash digest.
    """

    return hash_streaming(file_path)

def hash_data(file_path: Path) -> str:
    """Compute SHA-256 hash of persisted dataset contents.

    Args:
        file_path: Persisted dataset file path.

    Returns:
        str: SHA-256 hash digest.
    """

    return hash_streaming(file_path)

def hash_artifact(file_path: Path) -> str:
    """Compute SHA-256 hash of any persisted artifact file.

    Args:
        file_path: Artifact file path.

    Returns:
        str: SHA-256 hash digest.
    """

    return hash_streaming(file_path)

def hash_thresholds(thresholds: dict) -> str:
    """Compute deterministic hash for thresholds dictionary payload.

    Args:
        thresholds: Threshold mapping payload.

    Returns:
        str: Deterministic hash digest for thresholds payload.
    """

    if not thresholds.get("lineage", {}).get("created_at"):
        msg = "Missing 'created_at' timestamp in thresholds lineage. This timestamp is required for deterministic hashing of thresholds payloads. Please ensure that the 'created_at' field is included in the thresholds lineage metadata and is properly formatted as an ISO 8601 string."
        logger.error(msg)
        raise ConfigError(msg)
    thresholds["lineage"]["created_at"] = thresholds["lineage"]["created_at"].isoformat()

    return hash_dict(thresholds)
