"""Data loading utilities for assembling merged feature datasets from lineage."""

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from ml.exceptions import ConfigError, DataError
from ml.registry.hash_registry import HASH_LOADER_REGISTRY
from ml.utils.data.merge_dataset_into_main import merge_dataset_into_main
from ml.utils.data.models import DataLineageEntry
from ml.utils.loaders import read_data

logger = logging.getLogger(__name__)

required_fields = [
    "ref",
    "name",
    "version",
    "format",
    "path_suffix",
    "merge_key",
    "snapshot_id",
    "path",
    "loader_validation_hash",
    "data_hash",
    "row_count",
    "column_count",
]

def lineage_identity(entry: DataLineageEntry) -> tuple:
    """Return the identity tuple used for lineage consistency comparisons.

    Args:
        entry: Dataset lineage entry.

    Returns:
        tuple: Identity fields used to compare expected and actual lineage.
    """

    return (
        entry.name,
        entry.version,
        entry.snapshot_id,
        entry.data_hash,
        entry.loader_validation_hash,
        entry.merge_key,
    )

def load_and_validate_data(input_lineage: Iterable[DataLineageEntry]) -> pd.DataFrame:
    """Load, merge, hash-validate, and lineage-validate datasets from lineage entries.

    Args:
        input_lineage: Iterable of dataset lineage entries to load and merge.

    Returns:
        pd.DataFrame: Fully merged dataframe assembled from all lineage datasets.
    """

    input_lineage = list(input_lineage)
    data = pd.DataFrame()
    data_lineage = []

    if not input_lineage:
        msg = "No datasets specified in the input lineage."
        logger.error(msg)
        raise ConfigError(msg)
    
    for dataset in input_lineage:
        if dataset.format not in HASH_LOADER_REGISTRY:
            msg = f"Unsupported data format for loading and hashing: {dataset.format}"
            logger.error(msg)
            raise ConfigError(msg)
        
        dataset_path = Path(dataset.path)
        if not dataset_path.exists():
            msg = f"Dataset file not found at expected path: {dataset_path}"
            logger.error(msg)
            raise DataError(msg)
        df = read_data(dataset.format, dataset_path)
        merge_key = dataset.merge_key

        data, data_hash = merge_dataset_into_main(
            data=data,
            df=df,
            merge_key=merge_key,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            dataset_snapshot_path=Path(dataset.path).parent,
            dataset_path=dataset_path,
        )

        loader_validation_hash = HASH_LOADER_REGISTRY[dataset.format](dataset_path)
        
        data_lineage.append(DataLineageEntry(**{ 
            "ref": dataset.ref,
            "name": dataset.name,
            "version": dataset.version,
            "format": dataset.format,
            "path_suffix": dataset.path_suffix.format(format=dataset.format),
            "merge_key": dataset.merge_key,
            "snapshot_id": dataset.snapshot_id,
            "path": str(dataset_path),
            "loader_validation_hash": loader_validation_hash,
            "data_hash": data_hash,
            "row_count": len(df),
            "column_count": len(df.columns),
        }))

        logger.debug(f"Loaded dataset {dataset.name} {dataset.version} snapshot {dataset.snapshot_id} "
             f"file {dataset_path} loader hash {loader_validation_hash}")

    logger.info(f"Completed loading {len(input_lineage)} datasets. Final merged dataset shape: {data.shape}. Lineage: {data_lineage}")

    expected = {lineage_identity(e) for e in input_lineage}
    actual = {lineage_identity(e) for e in data_lineage}

    if expected != actual:
        msg = f"Data lineage mismatch after loading datasets. Expected lineage does not match actual lineage. Expected: {input_lineage}, Actual: {data_lineage}"
        logger.error(msg)
        raise DataError(msg)

    logger.debug("Data lineage validation passed. All expected datasets are present in the actual lineage after loading.")

    return data