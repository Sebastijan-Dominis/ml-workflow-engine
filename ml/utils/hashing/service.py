"""Hash function registries and convenience wrappers for artifacts/data.

The wrappers ``hash_file``, ``hash_data``, and ``hash_artifact`` are
intentional semantic aliases over the same streaming hash implementation.
They keep call sites domain-explicit while preserving a single hashing
mechanism.
"""

from collections.abc import Callable
from pathlib import Path

from ml.features.hashing.hash_arrow_metadata import hash_arrow_metadata
from ml.features.hashing.hash_parquet_metadata import hash_parquet_metadata
from ml.utils.hashing.hash_dict import hash_dict
from ml.utils.hashing.hash_streaming import hash_streaming

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

    return hash_dict(thresholds)
