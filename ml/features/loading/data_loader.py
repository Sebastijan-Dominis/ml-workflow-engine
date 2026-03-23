"""Data loading utilities for assembling merged feature datasets from lineage."""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pandas as pd

from ml.data.merge.merge_dataset_into_main import build_dataset_dag, merge_dataset_into_main
from ml.exceptions import ConfigError, DataError
from ml.feature_freezing.freeze_strategies.tabular.config.models import DatasetConfig
from ml.types import DataLineageEntry
from ml.utils.hashing.service import HASH_LOADER_REGISTRY
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
        entry.merge_how,
        entry.merge_validate,
    )

def load_and_validate_data(input_lineage: Iterable[DataLineageEntry]) -> pd.DataFrame:
    """Load, merge, hash-validate, and lineage-validate datasets from lineage entries
    in DAG-based topological order to respect merge dependencies.

    Args:
        input_lineage: Iterable of dataset lineage entries to load and merge.

    Returns:
        pd.DataFrame: Fully merged dataframe assembled from all lineage datasets.
    """

    input_lineage = list(input_lineage)
    if not input_lineage:
        msg = "No datasets specified in the input lineage."
        logger.error(msg)
        raise ConfigError(msg)

    # Build dataset dict and normalize merge_key
    dataset_dict = {}
    for ds in input_lineage:
        if ds.format not in HASH_LOADER_REGISTRY:
            msg = f"Unsupported data format for loading and hashing: {ds.format}"
            logger.error(msg)
            raise ConfigError(msg)

        # Normalize merge_key to list
        merge_key = ds.merge_key
        if isinstance(merge_key, str):
            merge_key = [merge_key]
        dataset_dict[ds.name] = (ds, merge_key)

    # Determine merge order via DAG
    datasets_for_dag = []
    for ds, mk in dataset_dict.values():
        # Prepare lightweight object for DAG: name + merge_key
        class _DagDataset:
            def __init__(self, name, merge_key):
                self.name = name
                self.merge_key = merge_key
        datasets_for_dag.append(_DagDataset(ds.name, mk))

    # datasets_for_dag is a lightweight runtime-only helper type; cast to the
    # declared DatasetConfig list type so mypy understands the call without
    # changing runtime behavior.
    merge_order = build_dataset_dag(cast(list[DatasetConfig], datasets_for_dag))

    data = pd.DataFrame()
    data_lineage: list[DataLineageEntry] = []

    for ds_name in merge_order:
        ds, merge_key = dataset_dict[ds_name]
        dataset_path = Path(ds.path.replace("\\", "/"))
        if not dataset_path.exists():
            msg = f"Dataset file not found at expected path: {dataset_path}"
            logger.error(msg)
            raise DataError(msg)

        df = read_data(ds.format, dataset_path)

        # Merge dataset into main DataFrame
        data, data_hash = merge_dataset_into_main(
            data=data,
            df=df,
            merge_key=merge_key,
            merge_how=ds.merge_how,
            merge_validate=ds.merge_validate,
            dataset_name=ds.name,
            dataset_version=ds.version,
            dataset_snapshot_path=Path(ds.path).parent,
            dataset_path=dataset_path,
        )

        loader_validation_hash = HASH_LOADER_REGISTRY[ds.format](dataset_path)

        # Append lineage
        data_lineage.append(DataLineageEntry(
            ref=ds.ref,
            name=ds.name,
            version=ds.version,
            format=ds.format,
            path_suffix=ds.path_suffix,
            merge_key=ds.merge_key,  # keep original type for lineage
            merge_how=ds.merge_how,
            merge_validate=ds.merge_validate,
            snapshot_id=ds.snapshot_id,
            path=str(dataset_path),
            loader_validation_hash=loader_validation_hash,
            data_hash=data_hash,
            row_count=len(df),
            column_count=len(df.columns),
        ))

    # Validate that the merged lineage matches expected
    expected = {lineage_identity(e) for e in input_lineage}
    actual = {lineage_identity(e) for e in data_lineage}
    if expected != actual:
        msg = f"Data lineage mismatch. Expected: {input_lineage}, Actual: {data_lineage}"
        logger.error(msg)
        raise DataError(msg)

    return data
