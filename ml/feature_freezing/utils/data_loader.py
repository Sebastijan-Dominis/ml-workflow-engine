"""Dataset loading and lineage assembly utilities for feature freezing."""

import logging
from pathlib import Path

import pandas as pd

from ml.data.merge.merge_dataset_into_main import merge_dataset_into_main
from ml.exceptions import ConfigError, DataError
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.snapshot_bindings.extraction.get_snapshot_binding import get_and_validate_snapshot_binding
from ml.types import DataLineageEntry
from ml.utils.hashing.service import HASH_LOADER_REGISTRY
from ml.utils.loaders import read_data
from ml.utils.snapshots.latest_snapshot import get_latest_snapshot_path

logger = logging.getLogger(__name__)

def load_data_with_lineage(
        config: TabularFeaturesConfig,
        snapshot_binding_key: str | None
    ) -> tuple[pd.DataFrame, list[DataLineageEntry]]:
    """Load configured datasets, merge them, and build lineage entries.

    Args:
        config: Validated tabular freeze configuration.

    Returns:
        tuple[pd.DataFrame, list[DataLineageEntry]]: Merged dataframe and
        dataset lineage metadata.
    """

    data = pd.DataFrame()
    data_lineage = []

    if not config.data:
        msg = "No datasets specified in the configuration."
        logger.error(msg)
        raise ConfigError(msg)

    datasets_binding = None
    if snapshot_binding_key:
        snapshot_binding_config = get_and_validate_snapshot_binding(
            snapshot_binding_key,
            expect_dataset_bindings=True
        )
        datasets_binding = snapshot_binding_config.datasets

    for dataset in config.data:
        dataset_binding = datasets_binding.get(dataset.name) if datasets_binding else None
        if dataset_binding:
            dataset_snapshot = dataset_binding.snapshot
            dataset_snapshot_path = Path(dataset.ref) / dataset.name / dataset_snapshot
        else:
            dataset_path = Path(dataset.ref) / dataset.name / dataset.version
            dataset_snapshot_path = get_latest_snapshot_path(dataset_path)

        if dataset.format not in HASH_LOADER_REGISTRY:
            msg = f"Unsupported data format for loading and hashing: {dataset.format}"
            logger.error(msg)
            raise ConfigError(msg)

        dataset_path = dataset_snapshot_path / dataset.path_suffix.format(format=dataset.format)
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
            dataset_snapshot_path=dataset_snapshot_path,
            dataset_path=dataset_path,
        )

        loader_validation_hash = HASH_LOADER_REGISTRY[dataset.format](dataset_path)

        entry = DataLineageEntry(
            ref=dataset.ref,
            name=dataset.name,
            version=dataset.version,
            format=dataset.format,
            path_suffix=dataset.path_suffix,
            merge_key=dataset.merge_key,
            snapshot_id=dataset_snapshot_path.name,
            path=dataset_path.as_posix(),
            loader_validation_hash=loader_validation_hash,
            data_hash=data_hash,
            row_count=len(df),
            column_count=len(df.columns),
        )

        data_lineage.append(entry)

        logger.debug(f"Loaded dataset {dataset.name} {dataset.version} snapshot {dataset_snapshot_path.name} "
             f"file {dataset_path} loader hash {loader_validation_hash}")

    logger.info(f"Completed loading {len(config.data)} dataframes. Final merged dataframe shape: {data.shape}. Lineage: {data_lineage}")

    return data, data_lineage
