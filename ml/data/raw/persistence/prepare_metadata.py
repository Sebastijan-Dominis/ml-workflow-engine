"""Metadata assembly helper for raw data snapshots."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml.data.utils.memory.get_memory_usage import get_memory_usage
from ml.exceptions import PersistenceError
from ml.utils.hashing.service import hash_data

logger = logging.getLogger(__name__)
    
def prepare_metadata(
    df: pd.DataFrame, 
    *, 
    args, 
    data_path: Path,
    raw_run_id: str,
    data_format: str,
    data_suffix: str
) -> dict:
    """Build metadata payload for a raw data snapshot.

    Args:
        df: Loaded raw dataframe.
        args: CLI arguments namespace with data/version/owner fields.
        data_path: Raw data file path.
        raw_run_id: Unique run identifier for raw handling stage.
        data_format: Raw data file format.
        data_suffix: Raw data file name or suffix.

    Returns:
        dict: Serializable metadata dictionary.
    """

    data_hash = hash_data(data_path)
        
    timestamp = datetime.now().isoformat()

    try:
        metadata = {
            "data": {
                "name": args.data,
                "version": args.version,
                "path_suffix": data_suffix,
                "format": data_format,
                "hash": data_hash
            },
            "rows": len(df),
            "columns": {
                "count": len(df.columns),
                "names": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict()
            },
            "created_at": timestamp,
            "created_by": "handle_raw.py",
            "owner": args.owner,
            "memory_usage_mb": get_memory_usage(df),
            "raw_run_id": raw_run_id
        }

        return metadata
    except Exception as e:
        msg = "Failed to prepare metadata for raw data"
        logger.error(msg + f": {str(e)}")
        raise PersistenceError(msg) from e