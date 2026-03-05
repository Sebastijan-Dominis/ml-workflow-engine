"""Metadata assembly helpers for processed data pipeline outputs."""

import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ml.config.compute_data_config_hash import compute_data_config_hash
from ml.data.config.schemas.processed import ProcessedConfig
from ml.data.processed.processing.add_row_id_base import RowIDMetadata
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.metadata.schemas.data.processed import ProcessedDatasetMetadata
from ml.metadata.validation.data.processed import validate_processed_dataset_metadata
from ml.utils.hashing.service import hash_data

logger = logging.getLogger(__name__)

def prepare_metadata(
    df: pd.DataFrame,
    *,
    config: ProcessedConfig,
    start_time: float,
    data_path: Path,
    source_data_path: Path,
    source_data_format: str,
    source_data_version: str,
    owner: str,
    memory_info: dict,
    processed_run_id: str,
    row_id_info: RowIDMetadata | None = None
) -> ProcessedDatasetMetadata:
    """Build metadata payload describing a processed data run.

    Args:
        df: Final processed dataframe.
        config: Validated processed configuration.
        start_time: Run start timestamp from ``time.perf_counter``.
        data_path: Persisted processed data file path.
        source_data_path: Source interim data file path.
        source_data_format: Source interim data format.
        source_data_version: Source interim data version.
        owner: Human owner for governance metadata.
        memory_info: Memory usage delta details.
        processed_run_id: Unique processed run identifier.
        row_id_info: Optional row-id generation trace metadata.

    Returns:
        ProcessedDatasetMetadata: Serializable metadata object.
    """

    data_hash = hash_data(data_path)

    config_hash = compute_data_config_hash(config)

    timestamp = iso_no_colon(datetime.now())

    duration = time.perf_counter() - start_time

    metadata_raw = {
        "processed_run_id": processed_run_id,
        "source_data": {
            "name": config.data.name,
            "snapshot_id": source_data_path.parent.name,
            "path": str(source_data_path),
            "format": source_data_format,
            "version": source_data_version,
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
        "created_by": "build_processed_dataset.py",
        "owner": owner,
        "duration": duration,
        "runtime_info": {
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "yaml_version": yaml.__version__,
            "python_version": platform.python_version(),
        }
    }

    # Keeps track of how row_id was generated for traceability and debugging purposes. Values for the same data should never change. Including this allows for detecting unexpected changes in code that generates row_id, which could lead to changes in row_id values and break lineage tracking. This is especially important for hotel_bookings where row_id is used for tracking guests across datasets.
    if row_id_info is not None:
        metadata_raw["row_id_info"] = row_id_info

    metadata = validate_processed_dataset_metadata(metadata_raw)
    logger.debug("Prepared metadata.")

    return metadata
