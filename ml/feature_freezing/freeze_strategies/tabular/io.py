import hashlib
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.exceptions import DataError, UserError
from ml.feature_freezing.utils.hashing import hash_streaming

logger = logging.getLogger(__name__)

def safe(val) -> str:
    return "None" if val is None else str(val)

def hash_parquet_metadata(path: Path) -> str:
    pf = pq.ParquetFile(path)
    meta = pf.metadata

    h = hashlib.sha256()

    for i in range(meta.num_columns):
        col = meta.schema.column(i)
        h.update(col.name.encode())
        h.update(safe(col.physical_type).encode())
        h.update(safe(col.logical_type).encode())

    h.update(safe(meta.num_rows).encode())
    h.update(safe(meta.created_by).encode())

    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        for j in range(rg.num_columns):
            col = rg.column(j)
            stats = col.statistics
            if stats:
                h.update(safe(stats.min).encode())
                h.update(safe(stats.max).encode())
                h.update(safe(stats.null_count).encode())
                h.update(safe(stats.distinct_count).encode())

    return h.hexdigest()

def hash_arrow_metadata(path: Path) -> str:
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

def load_and_hash_data(path: Path, format: str) -> tuple[pd.DataFrame, str]:
    HASH_REGISTRY = {
        "parquet": hash_parquet_metadata,
        "arrow": hash_arrow_metadata,
        "csv": hash_streaming,
        "json": hash_streaming,
    }

    if format not in HASH_REGISTRY:
        msg = f"Unsupported data format for loading and hashing: {format}"
        logger.error(msg)
        raise UserError(msg)

    data_hash = HASH_REGISTRY[format](path)

    FORMAT_REGISTRY = {
        "parquet": pd.read_parquet,
        "csv": pd.read_csv,
        "json": pd.read_json,
        "arrow": lambda p: pd.read_feather(p),
    }

    if format not in FORMAT_REGISTRY:
        msg = f"Unsupported data format for loading: {format}"
        logger.error(msg)
        raise UserError(msg)
    
    data = FORMAT_REGISTRY[format](path)
    return data, data_hash

def hash_feature_set(X: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for col in X.columns:
        h.update(col.encode())
        h.update(str(X[col].dtype).encode())
    return h.hexdigest()

def validate_feature_set_hashes_match(X: pd.DataFrame, expected_hash: str):
    actual_hash = hash_feature_set(X)
    if actual_hash != expected_hash:
        msg = f"Feature set hash mismatch: expected {expected_hash}, got {actual_hash}"
        logger.error(msg)
        raise DataError(msg)