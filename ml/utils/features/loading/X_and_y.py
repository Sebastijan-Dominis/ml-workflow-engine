import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import DataError
from ml.utils.features.hashing.hash_y import hash_y
from ml.utils.features.loading.resolve_feature_snapshots import resolve_feature_snapshots
from ml.utils.features.validation import validate_feature_set, validate_set
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

from ml.registry.format_registry import FORMAT_REGISTRY
from ml.registry.hash_registry import hash_file_streaming

def load_feature_set_data(snapshot_path: Path, fs, keys: list, strict: bool = True) -> tuple[pd.DataFrame, ...]:
    reader = FORMAT_REGISTRY.get(fs.data_format)
    if not reader:
        msg = f"Unsupported feature set format: {fs.data_format}"
        logger.error(msg)
        raise DataError(msg)
    
    metadata_path = snapshot_path / "metadata.json"
    metadata = load_json(metadata_path)

    data = []

    for key in keys:
        if not hasattr(fs, key):
            msg = f"Missing {key} in feature set specification."
            logger.error(msg)
            raise DataError(msg)
        file_path = snapshot_path / getattr(fs, key)
        df = reader(file_path)

        if strict:
            runtime_hash = hash_file_streaming(file_path)
            expected_hash = metadata.get("file_hashes", {}).get(key)
            
            if expected_hash is None:
                msg = f"No expected hash for {key} in metadata at {snapshot_path}."
                logger.error(msg)
                raise DataError(msg)
            
            if runtime_hash != expected_hash:
                msg = f"Hash mismatch for {key} in snapshot {snapshot_path}: expected {expected_hash}, got {runtime_hash}"
                logger.error(msg)
                raise DataError(msg)
        
        data.append(df)

    return tuple(data)

def load_X_and_y(model_cfg: SearchModelConfig | TrainModelConfig, keys: list[str], snapshot_selection: Optional[list[dict]], strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Load and combine X and y from multiple feature sets, validating lineage and data consistency.

    Args:
        model_cfg: The validated SearchModelConfig or TrainModelConfig
        keys: List of keys to load, e.g., ["X_train", "y_train"]
        snapshot_selection: Optional pre-resolved snapshot info from `resolve_feature_snapshots`. 
                            If None, the latest snapshots are automatically resolved.

    Returns:
        X: Combined feature DataFrame
        y: Target Series/DataFrame (assumes identical across feature sets)
        lineage: List of dicts with snapshot provenance for each feature set
    """
    feature_store_path = Path(model_cfg.feature_store.path)
    feature_sets = model_cfg.feature_store.feature_sets

    if not snapshot_selection:
        snapshot_selection = resolve_feature_snapshots(feature_store_path, feature_sets)

    dfs_X = []
    y_hashes = set()
    dataset_ids = set()
    schema_hashes = set()
    operator_hashes = set()
    feature_types = set()
    loader_validation_hashes = set()
    lineage = []

    for sel in snapshot_selection:
        fs = sel["fs_spec"]
        snapshot_path = sel["snapshot_path"]
        metadata = sel["metadata"]

        validate_feature_set(snapshot_path, metadata)
        
        X, y = load_feature_set_data(snapshot_path, fs, keys, strict=strict)
        
        dfs_X.append(X)

        y_hashes.add(hash_y(y))

        missing = [field for field in ["snapshot_identity_hash", "feature_schema_hash", "operators_hash", "feature_type", "loader_validation_hash", "file_hashes"] if metadata.get(field) is None]
        if missing:
            msg = f"Missing required metadata fields {missing} in snapshot {snapshot_path}"
            logger.error(msg)
            raise DataError(msg)
        
        dataset_id = metadata["snapshot_identity_hash"]
        feature_schema_hash = metadata["feature_schema_hash"]
        operators_hash = metadata["operators_hash"]
        feature_type = metadata["feature_type"]
        loader_validation_hash = metadata["loader_validation_hash"]
        file_hashes = metadata["file_hashes"]

        dataset_ids.add(dataset_id)
        schema_hashes.add(feature_schema_hash)
        operator_hashes.add(operators_hash)
        feature_types.add(feature_type)
        loader_validation_hashes.add(loader_validation_hash)

        lineage.append({
            "ref": fs.ref,
            "name": fs.name,
            "version": fs.version,
            "snapshot_id": snapshot_path.name,
            "snapshot_path": str(snapshot_path),
            "loader_validation_hash": loader_validation_hash,
            "file_hashes": file_hashes,
            "snapshot_identity_hash": dataset_id,
            "feature_schema_hash": feature_schema_hash,
            "operators_hash": operators_hash,
            "feature_type": feature_type,
        })

    validate_set("Target (y)", y_hashes, feature_sets)
    validate_set("Dataset identity", dataset_ids, feature_sets)
    validate_set("Feature schema", schema_hashes, feature_sets)
    validate_set("Feature operators", operator_hashes, feature_sets)
    validate_set("Feature type", feature_types, feature_sets)
    validate_set("Loader validation", loader_validation_hashes, feature_sets)

    for df in dfs_X[1:]:
        if not df.index.equals(dfs_X[0].index):
            msg = "Indices of feature sets do not match."
            logger.error(msg)
            raise DataError(msg)

    combined_df = pd.concat(dfs_X, axis=1)

    logger.debug(f"All validations for {len(snapshot_selection)} feature sets' {', '.join(keys)} passed. Combined shape: {combined_df.shape}")

    dupes = combined_df.columns[combined_df.columns.duplicated()]
    if len(dupes) > 0:
        logger.warning(f"Dropping duplicated columns: {list(dupes)}")
    
    X = combined_df.loc[:, ~combined_df.columns.duplicated()]

    return X, y, lineage