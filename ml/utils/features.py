import logging
logger = logging.getLogger(__name__)

import pandas as pd
import json
import hashlib
from datetime import datetime
from pathlib import Path

from ml.exceptions import DataError, PipelineContractError
from ml.config.validation_schemas.model_cfg import SearchModelConfig

def validate_feature_set(snapshot_path: Path, metadata: dict) -> None:
    if not metadata["data_hash"]:
        msg = f"Invalid or missing data hash in metadata at {snapshot_path}"
        logger.error(msg)
        raise DataError(msg)

def validate_set(hash_type: str, hashes: set, feature_sets: list) -> None:
    if len(hashes) != 1:
        msg = f"{hash_type} hashes do not match across feature sets. Feature sets involved: " + ", ".join(
            [f"{feature_sets[i].name} (version: {feature_sets[i].version})" for i in range(len(feature_sets))]
        )
        logger.error(msg)
        raise DataError(msg)

def hash_y(y) -> str:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            msg = f"hash_y only supports Series or single-column DataFrame, got {y.shape[1]} columns"
            logger.error(msg)
            raise DataError(msg)
        y = y.iloc[:, 0]

    if not isinstance(y, pd.Series):
        msg = f"hash_y expects a pandas Series or single-column DataFrame, got {type(y)}"
        logger.error(msg)
        raise DataError(msg)

    h = hashlib.sha256()

    # Numeric types
    if pd.api.types.is_numeric_dtype(y):
        # Use numpy array of floats to handle nullable dtypes safely
        arr = y.to_numpy(dtype="float64", copy=False)
        h.update(arr.tobytes())

    # Categorical types
    elif isinstance(y.dtype, pd.CategoricalDtype):
        h.update(y.cat.codes.to_numpy(dtype="int32", copy=False).tobytes())
        cat_bytes = b''.join(c.encode('utf-8') for c in y.cat.categories.astype(str))
        h.update(cat_bytes)

    # String/Object types
    elif pd.api.types.is_string_dtype(y) or pd.api.types.is_object_dtype(y):
        for val in y:
            h.update(str(val).encode('utf-8'))

    else:
        msg = f"Unsupported dtype for hashing: {y.dtype}"
        logger.error(msg)
        raise DataError(msg)

    return h.hexdigest()

def aggregate_schema_dfs(schemas: list[pd.DataFrame]) -> pd.DataFrame:
    if not schemas:
        return pd.DataFrame()

    aggregated_rows = []
    seen_features = set()

    for schema_df in schemas:
        if "feature" not in schema_df.columns:
            msg = "Schema must contain a 'feature' column"
            logger.error(msg)
            raise DataError(msg)

        for _, row in schema_df.iterrows():
            feature = row["feature"]
            if feature not in seen_features:
                seen_features.add(feature)
                aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows).reset_index(drop=True)

def get_latest_snapshot(version_path: Path) -> Path:
    snapshots = []
    
    for p in version_path.iterdir():
        if not p.is_dir():
            continue
        parts = p.name.split("_")
        if len(parts) != 2:
            logger.warning(f"Ignoring folder with unexpected format: {p.name}")
            continue
        timestamp_str, uuid_str = parts
        if 'T' not in timestamp_str:
            logger.warning(f"Ignoring folder with invalid timestamp: {p.name}")
            continue
        snapshots.append(p)
    
    if not snapshots:
        msg = f"No valid snapshots found in {version_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    def parse_snapshot(p: Path) -> tuple[datetime, str]:
        timestamp_str, uuid_str = p.name.split("_")
        date_part, time_part = timestamp_str.split('T')
        time_part = time_part.replace('-', ':')
        dt = datetime.fromisoformat(f"{date_part}T{time_part}")
        return (dt, uuid_str)

    latest_snapshot = max(snapshots, key=parse_snapshot)

    latest_dt, _ = parse_snapshot(latest_snapshot)
    tied_snapshots = [
        p for p in snapshots if parse_snapshot(p)[0] == latest_dt
    ]
    if len(tied_snapshots) > 1:
        tied_names = [p.name for p in tied_snapshots]
        logger.warning(
            f"Multiple snapshots have the same timestamp {latest_dt.isoformat()}: {tied_names}. "
            "Tie-breaking was done using UUIDs, which may not reflect true creation order."
        )

    return latest_snapshot

def get_metadata(latest_snapshot: Path) -> dict:
    metadata_path = latest_snapshot / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata

def get_cat_features(input_schema: pd.DataFrame, derived_schema: pd.DataFrame) -> list:
    input_categoricals = input_schema.loc[
        input_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    derived_categoricals = derived_schema.loc[
        derived_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    return input_categoricals + derived_categoricals

def load_feature_set_schemas(features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_schema_path = features_path / "input_schema.csv"
    derived_schema_path = features_path / "derived_schema.csv"
    try:
        input_schema = pd.read_csv(input_schema_path)
        derived_schema = pd.read_csv(derived_schema_path)
        return input_schema, derived_schema
    except Exception:
        logger.exception(f"Failed to load schemas from {features_path}.")
        raise

def load_schemas(model_cfg: SearchModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_store_path = Path(model_cfg.feature_store.path)
    feature_sets = model_cfg.feature_store.feature_sets

    if not feature_sets:
        msg = "No feature sets defined in model specifications."
        logger.error(msg)
        raise DataError(msg)

    input_schemas = []
    derived_schemas = []

    data_hashes = set()

    for fs in feature_sets:
        version_path = feature_store_path / fs.ref.replace(".", "/") / fs.name / fs.version
        latest_snapshot = get_latest_snapshot(version_path)
        metadata = get_metadata(latest_snapshot)

        validate_feature_set(latest_snapshot, metadata)

        curr_input_schema, curr_derived_schema = load_feature_set_schemas(version_path)
        input_schemas.append(curr_input_schema)
        derived_schemas.append(curr_derived_schema)

        data_hashes.add(metadata["data_hash"])

    validate_set("Data", data_hashes, feature_sets)
    input_schema = aggregate_schema_dfs(input_schemas)
    derived_schema = aggregate_schema_dfs(derived_schemas)

    return input_schema, derived_schema

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

def validate_model_feature_pipeline_contract(model_cfg: SearchModelConfig, pipeline_cfg: dict, cat_features: list | None = None) -> None:
    pipeline_supported_tasks = []
    if pipeline_cfg.get("assumptions", {}).get("supports_classification"):
        pipeline_supported_tasks.append("classification")
    if pipeline_cfg.get("assumptions", {}).get("supports_regression"):
        pipeline_supported_tasks.append("regression")

    if model_cfg.task.type not in pipeline_supported_tasks:
        msg = f"Pipeline does not support the task type: {model_cfg.task.type}"
        logger.error(msg)
        raise PipelineContractError(msg)
    
    if model_cfg.algorithm == "catboost":
        if cat_features is None:
            msg = "Categorical features must be provided for CatBoost models."
            logger.error(msg)
            raise PipelineContractError(msg)
        
        if not pipeline_cfg.get("assumptions", {}).get("handles_categoricals", False):
            msg = "Pipeline does not support categorical features required by CatBoost."
            logger.error(msg)
            raise PipelineContractError(msg)