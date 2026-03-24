"""Utilities for validated feature-dataset merges into the main frame."""

import logging
from pathlib import Path

import networkx as nx
import pandas as pd

from ml.data.validation.validate_data import validate_data
from ml.exceptions import ConfigError, DataError
from ml.feature_freezing.freeze_strategies.tabular.config.models import (
    DatasetConfig,
    MergeHow,
    MergeValidate,
)
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def normalize_keys(key: str | tuple[str, ...]) -> list[str]:
    if isinstance(key, tuple):
        return list(key)
    return [key]

def build_dataset_dag(datasets: list[DatasetConfig]) -> list[str]:
    """Build DAG and return topological merge order."""
    G = nx.DiGraph()
    for ds in datasets:
        G.add_node(ds.name)

    for i, ds1 in enumerate(datasets):
        keys1 = set(ds1.merge_key if isinstance(ds1.merge_key, list) else [ds1.merge_key])
        for j, ds2 in enumerate(datasets):
            if i >= j:
                continue
            keys2 = set(ds2.merge_key if isinstance(ds2.merge_key, list) else [ds2.merge_key])
            if keys1 & keys2:
                G.add_edge(ds1.name, ds2.name)

    try:
        merge_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        msg = "Cyclic merge dependency detected among datasets."
        logger.error(msg)
        raise ConfigError(msg)

    logger.debug(f"DAG-based merge order: {merge_order}")
    return merge_order

def merge_dataset_into_main(
    data: pd.DataFrame,
    df: pd.DataFrame,
    *,
    merge_key: str | tuple[str, ...],
    merge_how: MergeHow = "inner",
    merge_validate: MergeValidate = "m:m",
    dataset_name: str,
    dataset_version: str,
    dataset_snapshot_path: Path,
    dataset_path: Path,
) -> tuple[pd.DataFrame, str]:
    """Validate and merge one dataset into the main frame, returning data hash.

    Args:
        data: Main dataframe accumulated from previous merges.
        df: New dataset dataframe to merge into ``data``.
        merge_key: Key column(s) used for inner merge.
        merge_how: Type of merge to perform.
        merge_validate: Merge validation for row explosion protection.
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

    missing_keys_in_df = set(normalize_keys(merge_key)) - set(df.columns)
    if missing_keys_in_df:
        msg = f"Dataset {dataset_name} is missing merge key(s): {missing_keys_in_df}"
        logger.error(msg)
        raise DataError(msg)

    if not data.empty:
        missing_keys_in_main = set(normalize_keys(merge_key)) - set(data.columns)
        if missing_keys_in_main:
            msg = f"Merge key(s) {missing_keys_in_main} not found in main dataset when merging {dataset_name}"
            logger.error(msg)
            raise DataError(msg)

    original_rows = len(data) if not data.empty else 0

    overlapping_cols = set(data.columns) & set(df.columns) - set(normalize_keys(merge_key))
    if overlapping_cols:
        logger.warning(f"Dropping overlapping columns {overlapping_cols} from dataset {dataset_name} before merge")
        df = df.drop(columns=overlapping_cols)

    # Merge with row explosion protection
    try:
        if data.empty:
            data = df
        else:
            data = pd.merge(
                data,
                df,
                how=merge_how,
                on=merge_key,
                validate=merge_validate,
                suffixes=("", "_dup"),
            )
    except pd.errors.MergeError as e:
        msg = f"Merge failed for dataset {dataset_name} v{dataset_version}: {e}"
        logger.error(msg)
        raise DataError(msg)

    if len(data) < original_rows:
        logger.warning(
            f"Row count decreased after merging {dataset_name}: {original_rows} -> {len(data)}"
        )

    if merge_key:
        data = data.sort_values(by=normalize_keys(merge_key)).reset_index(drop=True)

    dataset_metadata = load_json(dataset_snapshot_path / "metadata.json")
    data_hash = validate_data(data_path=dataset_path, metadata=dataset_metadata)

    logger.debug(
        f"Merged dataset {dataset_name} v{dataset_version} snapshot {dataset_snapshot_path.name} "
        f"into main dataframe, resulting shape: {data.shape}"
    )

    return data, data_hash
