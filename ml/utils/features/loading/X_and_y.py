import logging
logger = logging.getLogger(__name__)
import pandas as pd
from pathlib import Path

from ml.exceptions import DataError
from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.utils.features.loading.metadata import get_metadata
from ml.utils.features.loading.latest_snapshot import get_latest_snapshot
from ml.utils.features.validation import validate_feature_set, validate_set
from ml.utils.features.hash_y import hash_y

FORMAT_REGISTRY = {
    "parquet": pd.read_parquet,
    "csv": pd.read_csv,
}

def load_feature_set_data(snapshot_path: Path, fs, keys: list) -> tuple[pd.DataFrame, ...]:
    reader = FORMAT_REGISTRY.get(fs.data_format)
    if not reader:
        msg = f"Unsupported feature set format: {fs.data_format}"
        logger.error(msg)
        raise DataError(msg)
    
    data = []

    for key in keys:
        if not hasattr(fs, key):
            msg = f"Missing {key} in feature set specification."
            logger.error(msg)
            raise DataError(msg)
        data.append(reader(Path(snapshot_path / getattr(fs, key))))

    return tuple(data)

def load_X_and_y(model_cfg: SearchModelConfig, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_store_path = Path(model_cfg.feature_store.path)
    feature_sets = model_cfg.feature_store.feature_sets

    if not feature_sets:
        msg = "No feature sets defined in model specifications."
        logger.error(msg)
        raise DataError(msg)

    dfs = []
    data_hashes = set()
    y_hashes = set()

    for fs in feature_sets:
        version_path = feature_store_path / fs.ref.replace(".", "/") / fs.name / fs.version
        latest_snapshot = get_latest_snapshot(version_path)
        metadata = get_metadata(latest_snapshot)
        
        validate_feature_set(latest_snapshot, metadata)
        
        X, y = load_feature_set_data(latest_snapshot, fs, keys)
        
        dfs.append(X)

        data_hashes.add(metadata.get("data_hash"))
        y_hashes.add(hash_y(y))

    validate_set("Data", data_hashes, feature_sets)
    validate_set("Target (y)", y_hashes, feature_sets)

    for df in dfs[1:]:
        if not df.index.equals(dfs[0].index):
            msg = "Indices of feature sets do not match."
            logger.error(msg)
            raise DataError(msg)

    combined_df = pd.concat(dfs, axis=1)

    dupes = combined_df.columns[combined_df.columns.duplicated()]
    if len(dupes) > 0:
        logger.warning(f"Dropping duplicated columns: {list(dupes)}")
    
    X = combined_df.loc[:, ~combined_df.columns.duplicated()]

    return X, y