from ml.utils.features.hashing.hash_streaming import hash_streaming
from ml.utils.features.hashing.hash_arrow_metadata import hash_arrow_metadata
from ml.utils.features.hashing.hash_parquet_metadata import hash_parquet_metadata


HASH_LOADER_REGISTRY = {
    "parquet": hash_parquet_metadata,
    "arrow": hash_arrow_metadata,
    "csv": hash_streaming,
    "json": hash_streaming,
}

def hash_file_streaming(file_path):
    """Compute SHA256 of file contents (streaming)."""
    return hash_streaming(file_path)