import hashlib
from pathlib import Path

from ml.utils.features.hashing.hash_arrow_metadata import hash_arrow_metadata
from ml.utils.features.hashing.hash_parquet_metadata import \
    hash_parquet_metadata
from ml.utils.hashing.hash_streaming import hash_streaming

HASH_LOADER_REGISTRY = {
    "parquet": hash_parquet_metadata,
    "arrow": hash_arrow_metadata,
    "csv": hash_streaming,
    "json": hash_streaming,
}

def hash_file_streaming(file_path: Path) -> str:
    """Compute SHA256 of file contents (streaming)."""
    return hash_streaming(file_path)

def hash_dataset(file_path: Path) -> str:
    """Compute SHA256 of file contents (streaming)."""
    return hash_streaming(file_path)

def hash_artifact(file_path: Path) -> str:
    """Compute SHA256 of file contents (streaming)."""
    return hash_streaming(file_path)

def hash_thresholds(thresholds: dict) -> str:
    """Compute SHA256 of thresholds dictionary."""
    return hashlib.sha256(str(thresholds).encode()).hexdigest()