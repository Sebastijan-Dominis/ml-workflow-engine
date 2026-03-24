"""Dataset loading and lineage assembly utilities for feature freezing."""

import logging
from pathlib import Path

import pandas as pd

from ml.data.merge.merge_dataset_into_main import build_dataset_dag, merge_dataset_into_main
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
    snapshot_binding_key: str | None = None
) -> tuple[pd.DataFrame, list[DataLineageEntry]]:
    """Load configured datasets, merge them, and build lineage entries.

    Args:
        config: Validated tabular freeze configuration.

    Returns:
        tuple[pd.DataFrame, list[DataLineageEntry]]: Merged dataframe and
        dataset lineage metadata.
    """

    if not config.data:
        msg = "No datasets specified in the configuration."
        logger.error(msg)
        raise ConfigError(msg)

    datasets_binding = None
    if snapshot_binding_key:
        snapshot_binding_config = get_and_validate_snapshot_binding(
            snapshot_binding_key, expect_dataset_bindings=True
        )
        datasets_binding = snapshot_binding_config.datasets

    dataset_dict = {ds.name: ds for ds in config.data}
    merge_order = build_dataset_dag(config.data)

    merged_data = pd.DataFrame()
    data_lineage: list[DataLineageEntry] = []

    for ds_name in merge_order:
        ds = dataset_dict[ds_name]
        dataset_versions = datasets_binding.get(ds.name) if datasets_binding else None
        dataset_snapshot_binding = dataset_versions.get(ds.version) if dataset_versions else None

        if snapshot_binding_key:
            if not dataset_snapshot_binding:
                msg = f"No snapshot binding found for {ds.name} {ds.version} under {snapshot_binding_key}"
                logger.error(msg)
                raise ConfigError(msg)
            dataset_snapshot = dataset_snapshot_binding.snapshot
            dataset_snapshot_path = Path(ds.ref) / ds.name / ds.version / dataset_snapshot
        else:
            dataset_path_base = Path(ds.ref) / ds.name / ds.version
            dataset_snapshot_path = get_latest_snapshot_path(dataset_path_base)

        dataset_path = dataset_snapshot_path / ds.path_suffix.format(format=ds.format)
        if not dataset_path.exists():
            msg = f"Dataset file not found at expected path: {dataset_path}"
            logger.error(msg)
            raise DataError(msg)
        df = read_data(ds.format, dataset_path)

        merged_data, data_hash = merge_dataset_into_main(
            data=merged_data,
            df=df,
            merge_key=tuple(ds.merge_key),
            merge_how=ds.merge_how,
            merge_validate=ds.merge_validate,
            dataset_name=ds.name,
            dataset_version=ds.version,
            dataset_snapshot_path=dataset_snapshot_path,
            dataset_path=dataset_path,
        )

        loader_validation_hash = HASH_LOADER_REGISTRY[ds.format](dataset_path)

        entry = DataLineageEntry(
            ref=ds.ref,
            name=ds.name,
            version=ds.version,
            format=ds.format,
            path_suffix=ds.path_suffix,
            merge_key=tuple(ds.merge_key),
            merge_how=ds.merge_how,
            merge_validate=ds.merge_validate,
            snapshot_id=dataset_snapshot_path.name,
            path=dataset_path.as_posix(),
            loader_validation_hash=loader_validation_hash,
            data_hash=data_hash,
            row_count=len(df),
            column_count=len(df.columns),
        )
        data_lineage.append(entry)
        logger.debug(f"Loaded dataset {ds.name} {ds.version} snapshot {dataset_snapshot_path.name} loader hash {loader_validation_hash}")

    logger.info(f"Completed loading {len(config.data)} datasets. Final merged dataframe shape: {merged_data.shape}.")
    return merged_data, data_lineage
