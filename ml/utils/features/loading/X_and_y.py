import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.config.validation_schemas.model_cfg import (SearchModelConfig,
                                                    TrainModelConfig)
from ml.exceptions import DataError
from ml.utils.data.loader import load_data_with_loader_validation_hash
from ml.utils.data.validate_dataset import validate_dataset
from ml.utils.data.validate_row_id import validate_row_id
from ml.utils.features.loading.get_target import get_target
from ml.utils.features.loading.resolve_feature_snapshots import \
    resolve_feature_snapshots
from ml.utils.features.segmentation.segment import apply_segmentation
from ml.utils.features.validation.validate_set import validate_set
from ml.utils.features.validation.validate_target import validate_target
from ml.utils.features.validation.validate_feature_set import (
    ensure_required_fields_present, validate_feature_set)
from ml.utils.loaders import load_json, read_data

logger = logging.getLogger(__name__)

def load_X_and_y(
    model_cfg: SearchModelConfig | TrainModelConfig, 
    *,
    snapshot_selection: Optional[list[dict]], 
    drop_row_id: bool = True,
    strict: bool = True
) -> tuple[pd.DataFrame, pd.Series, list[dict]]:
    segmented_df: pd.DataFrame
    y: pd.Series

    feature_store_path = Path(model_cfg.feature_store.path)
    feature_sets = model_cfg.feature_store.feature_sets

    if not snapshot_selection:
        snapshot_selection = resolve_feature_snapshots(feature_store_path, feature_sets)

    dfs_X = []
    feature_types = set()
    loader_validation_hashes = set()
    lineage = []

    for sel in snapshot_selection:
        fs = sel["fs_spec"]
        snapshot_path = sel["snapshot_path"]
        metadata = sel["metadata"]

        ensure_required_fields_present(snapshot_path, metadata)
        
        file_path = snapshot_path / getattr(fs, "file_name")

        df_X = read_data(fs.data_format, file_path)
        
        validate_feature_set(
            feature_set=df_X, 
            metadata=metadata, 
            file_path=file_path, 
            strict=strict
        )

        dfs_X.append(df_X)
        
        feature_schema_hash = metadata["feature_schema_hash"]
        operators_hash = metadata["operators_hash"]
        feature_type = metadata["feature_type"]
        loader_validation_hash = metadata["loader_validation_hash"]
        file_hash = metadata["file_hash"]
        in_memory_hash = metadata["in_memory_hash"]

        feature_types.add(feature_type)
        loader_validation_hashes.add(loader_validation_hash)

        lineage.append({
            "name": fs.name,
            "version": fs.version,
            "snapshot_id": snapshot_path.name,
            "snapshot_path": str(snapshot_path),
            "loader_validation_hash": loader_validation_hash,
            "file_hash": file_hash,
            "in_memory_hash": in_memory_hash,
            "feature_schema_hash": feature_schema_hash,
            "operators_hash": operators_hash,
            "feature_type": feature_type,
        })

    validate_set("Feature type", feature_types, feature_sets)

    dataset, loader_validation_hash = load_data_with_loader_validation_hash(
        path=Path(model_cfg.data.path),
        format=model_cfg.data.format
    )
    data_metadata = load_json(path=Path(model_cfg.data.metadata_path))
    validate_dataset(data_path=Path(model_cfg.data.path), metadata=data_metadata)
    loader_validation_hashes.add(loader_validation_hash)
    validate_set("Loader validation", loader_validation_hashes, feature_sets)

    target_name = model_cfg.target.name
    target_version = model_cfg.target.version
    key = (target_name, target_version)
    y = get_target(data=dataset, key=key)

    for df in dfs_X:
        validate_row_id(df)

    for df in dfs_X[1:]:
        if not df.index.equals(dfs_X[0].index):
            msg = "Indices of feature sets do not match."
            logger.error(msg)
            raise DataError(msg)

    merged_df = dfs_X[0]
    if len(dfs_X) > 1:
        for df in dfs_X[1:]:
            merged_df = merged_df.merge(df, on="row_id", how="inner", suffixes=("", "_dup"))

    cols = merged_df.columns.tolist()
    if "row_id" in cols:
        cols.insert(0, cols.pop(cols.index("row_id")))
    merged_df = merged_df[cols]

    if target_name in merged_df.columns:
        try:
            merged_df = merged_df.drop(columns=[target_name], axis=1, errors="raise")
        except KeyError:
            msg = f"Target column {target_name} not found in merged feature set, but was expected."
            logger.error(msg)
            raise DataError(msg)

    logger.debug(f"All validations for {len(snapshot_selection)} feature sets' passed. Combined shape: {merged_df.shape}")

    segmented_df = apply_segmentation(
        data=merged_df,
        seg_cfg=model_cfg.segmentation
    )

    logger.debug(f"Applied the segmentation step. Resulting shape: {segmented_df.shape}")

    dupes = segmented_df.columns[segmented_df.columns.duplicated()]
    if len(dupes) > 0:
        logger.warning(f"Dropping duplicated columns: {list(dupes)}")
        segmented_df = segmented_df.drop(columns=dupes, axis=1)
    X = segmented_df.loc[:, ~segmented_df.columns.duplicated()].copy()

    full_df = pd.concat([X, y], axis=1)
    validate_target(y=y, tgt_cfg=model_cfg.target, data=full_df)
    logger.debug("Target validation passed.")

    if drop_row_id:
        if "row_id" not in X.columns:
            msg = "Cannot drop 'row_id' column because it is not present in the features."
            logger.error(msg)
            raise DataError(msg)
        X = X.drop(columns=["row_id"])
        logger.debug("Dropped 'row_id' column from features as requested.")

    logger.info(f"Successfully loaded features and target. Final shapes - X: {X.shape}, y: {y.shape}")
    
    return X, y, lineage