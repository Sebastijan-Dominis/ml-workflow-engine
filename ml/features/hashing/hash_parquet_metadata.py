"""Hashing helpers for Parquet metadata integrity fingerprints."""

import hashlib
import logging
from pathlib import Path

import pyarrow.parquet as pq
from ml.exceptions import RuntimeMLError
from ml.features.hashing.safe import safe

logger = logging.getLogger(__name__)

def hash_parquet_metadata(path: Path) -> str:
    """Compute a deterministic hash from Parquet schema and statistics metadata.

    Args:
        path: Path to the Parquet file to fingerprint.

    Returns:
        Deterministic metadata hash for the Parquet file.
    """

    try:
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
    except Exception as e:
        msg = f"Error hashing Parquet metadata for file {path}. "
        logger.error(msg + f"Details: {str(e)}")
        raise RuntimeMLError(msg) from e
