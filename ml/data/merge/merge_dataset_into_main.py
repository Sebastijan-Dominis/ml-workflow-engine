"""Utilities for validated feature-dataset merges into the main frame."""

import logging
from pathlib import Path

import pandas as pd
from ml.data.validation.validate_data import validate_data
from ml.exceptions import DataError
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def merge_dataset_into_main(
    data: pd.DataFrame,
    df: pd.DataFrame,
    *,
    merge_key: str,
    dataset_name: str,
    dataset_version: str,
    dataset_snapshot_path: Path,
    dataset_path: Path,
) -> tuple[pd.DataFrame, str]:
    """Validate and merge one dataset into the main frame, returning data hash.

    Args:
        data: Main dataframe accumulated from previous merges.
        df: New dataset dataframe to merge into ``data``.
        merge_key: Key column used for inner merge.
        dataset_name: Logical dataset name used for logging/errors.
        dataset_version: Dataset version string.
        dataset_snapshot_path: Snapshot directory containing dataset metadata.
        dataset_path: Dataset file path used for data validation.

    Returns:
        Tuple of merged dataframe and validated data hash.

    Raises:
        DataError: If merge keys are missing, merge alignment fails, or merged
            output becomes empty.

    Side Effects:
        May drop overlapping non-key columns from ``df`` before merge and logs
        dataset-level merge diagnostics/warnings.
    """

    if merge_key not in df.columns:
        msg = f"Dataset {dataset_name} is missing merge key '{merge_key}' column."
        logger.error(msg)
        raise DataError(msg)
    if merge_key not in data.columns and not data.empty:
        msg = f"Merge key '{merge_key}' column from dataset {dataset_name} not found in the main dataset for merging."
        logger.error(msg)
        raise DataError(msg)

    dataset_metadata = load_json(dataset_snapshot_path / "metadata.json")
    data_hash = validate_data(data_path=dataset_path, metadata=dataset_metadata)

    logger.debug(f"Starting to merge dataset {dataset_name} v{dataset_version} snapshot {dataset_snapshot_path.name} with shape {df.shape} into the main dataset with shape {data.shape}")

    if data.empty:
        data = df
    else:
        overlapping_cols = set(data.columns) & set(df.columns) - {merge_key}
        if overlapping_cols:
            logger.warning(f"Overlapping columns found in dataset {dataset_name}: {overlapping_cols}. Dropping these columns from the new dataset before merge.")
            df = df.drop(columns=overlapping_cols)
        data = pd.merge(data, df, how="inner", on=merge_key)
        if data.empty:
            msg = f"Merged dataset is empty after merging with {dataset_name}. Please check the '{merge_key}' alignment across datasets."
            logger.error(msg)
            raise DataError(msg)

    return data, data_hash
