"""Orchestration utilities for loading, validating, and aligning features and targets."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.config.schemas.model_cfg import (SearchModelConfig,
                                                    TrainModelConfig)
from ml.exceptions import DataError
from ml.utils.data.models import DataLineageEntry
from ml.utils.data.validate_min_rows import validate_min_rows
from ml.utils.data.validate_row_id import validate_row_id
from ml.utils.features.loading.data_loader import load_and_validate_data
from ml.utils.features.loading.get_target import get_target_with_row_id
from ml.utils.features.loading.resolve_feature_snapshots import \
    resolve_feature_snapshots
from ml.utils.features.segmentation.segment import apply_segmentation
from ml.utils.features.validation.validate_feature_set import \
    validate_feature_set
from ml.utils.features.validation.validate_feature_target_row_id import \
    validate_feature_target_row_id
from ml.utils.features.validation.validate_set import validate_set
from ml.utils.features.validation.validate_target import validate_target
from ml.utils.loaders import read_data
from ml.utils.validate_dict import ensure_required_fields_present_in_dict

logger = logging.getLogger(__name__)

def load_X_and_y(
    model_cfg: SearchModelConfig | TrainModelConfig, 
    *,
    snapshot_selection: Optional[list[dict]], 
    drop_row_id: bool = True,
    strict: bool = True
) -> tuple[pd.DataFrame, pd.Series, list[dict]]:
    """Load feature sets and target, apply validations, and return `(X, y, lineage)`.

    Args:
        model_cfg: Validated model configuration driving data loading and validation behavior.
        snapshot_selection: Optional pre-resolved snapshot selection descriptors.
        drop_row_id: Whether to drop `row_id` from output features.
        strict: Whether strict hash/integrity checks should be enforced.

    Returns:
        tuple[pd.DataFrame, pd.Series, list[dict]]: Features, target series, and feature-lineage metadata.

    Raises:
        DataError: If feature/target integrity checks fail, required metadata is
            invalid, or segmentation/alignment leaves invalid data.

    Notes:
        When ``snapshot_selection`` is not supplied, snapshots are resolved from
        the feature-store config and then validated for schema, hashing, lineage,
        and row alignment before target extraction.

    Side Effects:
        Performs disk reads across feature snapshots/metadata and emits extensive
        validation logs.
    """

    segmented_df: pd.DataFrame
    y_with_row_id: pd.DataFrame
    y: pd.Series

    feature_store_path = Path(model_cfg.feature_store.path)
    feature_sets = model_cfg.feature_store.feature_sets

    if not snapshot_selection:
        snapshot_selection = resolve_feature_snapshots(feature_store_path, feature_sets)

    dfs_X = []
    feature_types = set()
    data_lineage: set[DataLineageEntry] = set()
    feature_lineage = []

    required_fields = [
        "feature_schema_hash",
        "operators_hash",
        "feature_type",
        "data_lineage",
        "in_memory_hash",
        "file_hash",
    ]

    for sel in snapshot_selection:
        fs = sel["fs_spec"]
        snapshot_path = sel["snapshot_path"]
        metadata = sel["metadata"]

        logger.debug(f"Verifying required metadata fields for feature set {fs.name} v{fs.version} snapshot {snapshot_path.name}")
        ensure_required_fields_present_in_dict(
            input_dict=metadata, 
            required_fields=required_fields
        )
        
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
        file_hash = metadata["file_hash"]
        in_memory_hash = metadata["in_memory_hash"]
        for dataset_dict in metadata["data_lineage"]:
            try:
                entry = DataLineageEntry(**dataset_dict)
            except TypeError as e:
                msg = f"Data lineage entry is missing required fields or has extra fields. Dataset dict: {dataset_dict}. Error: {e}"
                logger.error(msg)
                raise DataError(msg)
            data_lineage.add(entry)

        feature_types.add(feature_type)

        feature_lineage.append({
            "name": fs.name,
            "version": fs.version,
            "snapshot_id": snapshot_path.name,
            "snapshot_path": str(snapshot_path),
            "file_hash": file_hash,
            "in_memory_hash": in_memory_hash,
            "feature_schema_hash": feature_schema_hash,
            "operators_hash": operators_hash,
            "feature_type": feature_type,
        })

    validate_set("Feature type", feature_types, feature_sets)

    data = load_and_validate_data(data_lineage)

    target_name = model_cfg.target.name
    target_version = model_cfg.target.version
    key = (target_name, target_version)
    y_with_row_id = get_target_with_row_id(data=data, key=key)

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

    dupes = segmented_df.columns[segmented_df.columns.duplicated()]
    if len(dupes) > 0:
        logger.warning(f"Dropping duplicated columns: {list(dupes)}")
        segmented_df = segmented_df.drop(columns=dupes, axis=1)
    X = segmented_df.loc[:, ~segmented_df.columns.duplicated()].copy()

    validate_feature_target_row_id(X=X, y_with_row_id=y_with_row_id)

    validate_min_rows(X, model_cfg.min_rows)

    full_df = X.merge(y_with_row_id, on="row_id", how="inner", suffixes=("", "_target"))
    
    y = full_df[target_name].copy()

    validate_target(
        y=y, 
        model_cfg=model_cfg, 
        data=full_df
    )

    if drop_row_id:
        if "row_id" not in X.columns:
            msg = "Cannot drop 'row_id' column because it is not present in the features."
            logger.error(msg)
            raise DataError(msg)
        X = X.drop(columns=["row_id"])
        logger.debug("Dropped 'row_id' column from features as requested.")

    logger.info(f"Successfully loaded features and target. Final shapes - X: {X.shape}, y: {y.shape}")
    
    return X, y, feature_lineage