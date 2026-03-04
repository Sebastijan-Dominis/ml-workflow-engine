"""Metadata assembly helpers for interim data pipeline outputs."""

import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.config.compute_data_config_hash import compute_data_config_hash
from ml.data.config.schemas.interim import InterimConfig
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.metadata.schemas.data.interim import InterimDatasetMetadata
from ml.metadata.validation.data.interim import validate_interim_dataset_metadata
from ml.utils.hashing.service import hash_data

logger = logging.getLogger(__name__)

def prepare_metadata(
    df: pd.DataFrame, 
    *, 
    config: InterimConfig, 
    start_time: float, 
    data_path: Path,
    source_data_path: Path, 
    source_data_format: str,
    owner: str, 
    memory_info: dict,
    interim_run_id: str
) -> InterimDatasetMetadata:
    """Build metadata payload describing an interim data run.

    Args:
        df: Final interim dataframe.
        config: Validated interim configuration.
        start_time: Run start timestamp from ``time.perf_counter``.
        data_path: Persisted interim data file path.
        source_data_path: Source raw data file path.
        source_data_format: Source raw data format.
        owner: Human owner for governance metadata.
        memory_info: Memory usage delta details.
        interim_run_id: Unique interim run identifier.

    Returns:
        InterimDatasetMetadata: The validated metadata object.
    """

    data_hash = hash_data(data_path)
         
    config_hash = compute_data_config_hash(config)
    
    timestamp = iso_no_colon(datetime.now())

    duration = time.perf_counter() - start_time

    metadata_raw = {
        "interim_run_id": interim_run_id,
        "source_data": {
            "name": config.data.name,
            "snapshot_id": source_data_path.parent.name,
            "path": str(source_data_path),
            "format": source_data_format,
            "version": config.raw_data_version,
        },
        "data": {
            "name": config.data.name,
            "version": config.data.version,
            "output": {
                "path_suffix": config.data.output.path_suffix,
                "format": config.data.output.format,
                "compression": config.data.output.compression,
            },
            "hash": data_hash,
        },
        "memory": memory_info,
        "rows": len(df),
        "columns": {
            "count": len(df.columns),
            "names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict()
        },
        "config_hash": config_hash,
        "created_at": timestamp,
        "created_by": "build_interim_dataset.py",
        "owner": owner,
        "duration": duration,
        "runtime_info": {
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "yaml_version": yaml.__version__,
            "python_version": platform.python_version(),
        }
    }

    metadata = validate_interim_dataset_metadata(metadata_raw)
    logger.debug(f"Prepared metadata.")

    return metadata