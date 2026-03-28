"""Orchestration utilities for loading, validating, and aligning features and targets."""

import logging
from pathlib import Path

import pandas as pd

from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.data.validation.validate_entity_key import validate_entity_key
from ml.data.validation.validate_min_rows import validate_min_rows
from ml.exceptions import DataError
from ml.features.loading.data_loader import load_and_validate_data
from ml.features.loading.get_target import get_target_with_entity_key
from ml.features.loading.resolve_feature_snapshots import resolve_feature_snapshots
from ml.features.segmentation.segment import apply_segmentation
from ml.features.validation.validate_feature_set import validate_feature_set
from ml.features.validation.validate_feature_target_entity_key import (
    validate_feature_target_entity_key,
)
from ml.features.validation.validate_set import validate_set
from ml.features.validation.validate_target import validate_target
from ml.io.validation.validate_mapping import ensure_required_fields_present_in_dict
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.modeling.validation.feature_lineage import validate_and_construct_feature_lineage
from ml.types import DataLineageEntry
from ml.utils.loaders import read_data

logger = logging.getLogger(__name__)

# Arbitrary threshold for flagging significant drops in coverage during merges, which may indicate misalignment issues. Hardcoded here for now but could be made configurable in the future if needed. Note that this is not a strict validation failure, because some drop in coverage may be expected when merging multiple feature sets, but significant drops may warrant investigation.
COVERAGE_WARNING_THRESHOLD = 0.9

def load_features_and_target(
    model_cfg: SearchModelConfig | TrainModelConfig,
    *,
    snapshot_selection: list[dict] | None,
    snapshot_binding_key: str | None = None,
    drop_entity_key: bool = True,
    strict: bool = True
) -> tuple[pd.DataFrame, pd.Series, list[FeatureLineage], str]:
    """Load feature sets and target, apply validations, and return `(X, y, lineage)`.

    Args:
        model_cfg: Validated model configuration driving data loading and validation behavior.
        snapshot_selection: Optional pre-resolved snapshot selection descriptors.
        snapshot_binding_key: Optional key for a snapshot binding to define which snapshot to load for each dataset.
        drop_entity_key: Whether to drop `entity_key` from output features.
        strict: Whether strict hash/integrity checks should be enforced.

    Returns:
        tuple[pd.DataFrame, pd.Series, list[FeatureLineage], str]: Features, target series, feature lineage information, and entity key name.

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
    y_with_entity_key: pd.DataFrame
    y: pd.Series

    feature_store_path = Path(model_cfg.feature_store.path)
    feature_sets = model_cfg.feature_store.feature_sets

    if not snapshot_selection:
        snapshot_selection = resolve_feature_snapshots(
            feature_store_path,
            feature_sets,
            snapshot_binding_key=snapshot_binding_key
        )

    feature_types = set()
    data_lineage: set[DataLineageEntry] = set()
    feature_lineage_raw = []
    fs_dict = {}
    entity_keys = set()

    required_fields = [
        "feature_schema_hash",
        "operator_hash",
        "feature_type",
        "data_lineage",
        "in_memory_hash",
        "file_hash",
        "entity_key",
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

        file_path = snapshot_path / fs.file_name

        df_X = read_data(fs.data_format, file_path)

        entity_key_curr = metadata["entity_key"]
        validate_entity_key(df_X, entity_key_curr)
        if not isinstance(entity_key_curr, str):
            msg = f"entity_key must be a string, got {entity_key_curr} of type {type(entity_key_curr)} in feature set {fs.name} v{fs.version} snapshot {snapshot_path.name}"
            logger.error(msg)
            raise DataError(msg)

        entity_keys.add(entity_key_curr)

        validate_feature_set(
            feature_set=df_X,
            metadata=metadata,
            file_path=file_path,
            strict=strict
        )

        fs_name = sel["fs_spec"].name
        fs_dict[fs_name] = {
            "df": df_X,
            "sel": sel
        }

        feature_schema_hash = metadata["feature_schema_hash"]
        operator_hash = metadata["operator_hash"]
        feature_type = metadata["feature_type"]
        file_hash = metadata["file_hash"]
        in_memory_hash = metadata["in_memory_hash"]
        for dataset_dict in metadata["data_lineage"]:
            try:
                entry = DataLineageEntry(**dataset_dict)
            except TypeError as e:
                msg = f"Data lineage entry is missing required fields or has extra fields. Dataset dict: {dataset_dict}."
                logger.exception(msg)
                raise DataError(msg) from e
            data_lineage.add(entry)

        feature_types.add(feature_type)

        feature_lineage_raw.append({
            "name": fs.name,
            "version": fs.version,
            "snapshot_id": snapshot_path.name,
            "snapshot_path": str(snapshot_path),
            "file_hash": file_hash,
            "in_memory_hash": in_memory_hash,
            "feature_schema_hash": feature_schema_hash,
            "operator_hash": operator_hash,
            "feature_type": feature_type,
            "entity_key": entity_key_curr,
            "file_name": fs.file_name,
            "data_format": fs.data_format,
        })

    validate_set("Feature type", feature_types, feature_sets)
    if len(entity_keys) > 1:
        msg = f"Multiple entity keys found across feature sets: {entity_keys}. All feature sets must share the same entity key for alignment."
        logger.error(msg)
        raise DataError(msg)
    entity_key = entity_keys.pop()

    data = load_and_validate_data(data_lineage)

    target_name = model_cfg.target.name
    target_version = model_cfg.target.version
    key = (target_name, target_version)
    y_with_entity_key = get_target_with_entity_key(data=data, key=key, entity_key=entity_key)

    merge_order = sorted(fs_dict.keys())

    merged_df = pd.DataFrame()

    for fs_name in merge_order:
        df = fs_dict[fs_name]["df"]

        previous_len = len(merged_df) if not merged_df.empty else None

        if merged_df.empty:
            merged_df = df
        else:
            overlapping_cols = set(merged_df.columns) & set(df.columns) - {entity_key}
            if overlapping_cols:
                logger.warning(
                    f"Dropping overlapping columns {overlapping_cols} from feature set {fs_name}"
                )
                df = df.drop(columns=overlapping_cols)

            merged_df = merged_df.merge(
                df,
                on=entity_key,
                how="inner",
                validate="1:1",
                suffixes=("", "_dup"),
            )

            if previous_len is not None and previous_len != len(merged_df):
                coverage = len(merged_df)/previous_len
                if coverage < COVERAGE_WARNING_THRESHOLD:
                    logger.warning(f"Merging {fs_name} reduced rows from {previous_len} -> {len(merged_df)} ({coverage:.2%})")

    merged_df = merged_df.sort_values(by=entity_key).reset_index(drop=True)

    cols = merged_df.columns.tolist()
    if entity_key in cols:
        cols.insert(0, cols.pop(cols.index(entity_key)))
    merged_df = merged_df[cols]

    if target_name in merged_df.columns:
        try:
            merged_df = merged_df.drop(columns=[target_name], axis=1, errors="raise")
        except KeyError as e:
            msg = f"Target column {target_name} not found in merged feature set, but was expected."
            logger.exception(msg)
            raise DataError(msg) from e

    logger.debug(f"All validations for {len(snapshot_selection)} feature sets' passed. Combined shape: {merged_df.shape}")

    segmented_df = apply_segmentation(
        data=merged_df,
        seg_cfg=model_cfg.segmentation
    )

    dupes = segmented_df.columns[segmented_df.columns.duplicated()]
    if len(dupes) > 0:
        logger.warning(
            "Dropping duplicate column occurrences (keeping first occurrence): %s",
            list(dupes),
        )
    X = segmented_df.loc[:, ~segmented_df.columns.duplicated()].copy()

    validate_feature_target_entity_key(X=X, y_with_entity_key=y_with_entity_key, entity_key=entity_key)

    full_df = X.merge(y_with_entity_key, on=entity_key, how="inner", validate="1:1", suffixes=("", "_target"))

    y = full_df[target_name].copy()

    validate_min_rows(full_df, model_cfg.min_rows)

    validate_target(
        y=y,
        model_cfg=model_cfg,
        data=full_df
    )

    if drop_entity_key:
        if entity_key not in X.columns:
            msg = f"Cannot drop '{entity_key}' column because it is not present in the features."
            logger.error(msg)
            raise DataError(msg) from None
        X = X.drop(columns=[entity_key])
        logger.debug(f"Dropped '{entity_key}' column from features as requested.")

    feature_lineage = validate_and_construct_feature_lineage(feature_lineage_raw)

    logger.info(f"Successfully loaded features and target. Final shapes - X: {X.shape}, y: {y.shape}")

    return X, y, feature_lineage, entity_key
