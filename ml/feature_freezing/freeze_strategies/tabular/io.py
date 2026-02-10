import hashlib
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.exceptions import DataError, UserError
from ml.registry.format_registry import FORMAT_REGISTRY
from ml.registry.hash_registry import HASH_LOADER_REGISTRY

logger = logging.getLogger(__name__)

def load_data_with_loader_validation_hash(path: Path, format: str) -> tuple[pd.DataFrame, str]:
    if format not in HASH_LOADER_REGISTRY:
        msg = f"Unsupported data format for loading and hashing: {format}"
        logger.error(msg)
        raise UserError(msg)

    loader_validation_hash = HASH_LOADER_REGISTRY[format](path)

    if format not in FORMAT_REGISTRY:
        msg = f"Unsupported data format for loading: {format}"
        logger.error(msg)
        raise UserError(msg)
    
    data = FORMAT_REGISTRY[format](path)
    return data, loader_validation_hash

def hash_feature_schema(X: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for col in X.columns:
        h.update(col.encode())
        h.update(str(X[col].dtype).encode())
    return h.hexdigest()

def validate_feature_schema_hashes_match(X: pd.DataFrame, expected_hash: str):
    actual_hash = hash_feature_schema(X)
    if actual_hash != expected_hash:
        msg = f"Feature set hash mismatch: expected {expected_hash}, got {actual_hash}"
        logger.error(msg)
        raise DataError(msg)