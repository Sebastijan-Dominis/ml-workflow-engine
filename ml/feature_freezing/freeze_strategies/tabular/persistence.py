import logging
from pathlib import Path

import pandas as pd

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.registry.feature_operators import FEATURE_OPERATORS

logger = logging.getLogger(__name__)

def freeze_parquet(path: Path, *, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame, compression=None) -> dict:
    X_train.to_parquet(path / "X_train.parquet", index=False, compression=compression)
    X_val.to_parquet(path / "X_val.parquet", index=False, compression=compression)
    X_test.to_parquet(path / "X_test.parquet", index=False, compression=compression)

    y_train.to_parquet(path / "y_train.parquet", index=False, compression=compression)
    y_val.to_parquet(path / "y_val.parquet", index=False, compression=compression)
    y_test.to_parquet(path / "y_test.parquet", index=False, compression=compression)    
    
    logger.info(f"Tabular features saved to {path}")

    data_paths = {
        "X_train": path / "X_train.parquet",
        "X_val": path / "X_val.parquet",
        "X_test": path / "X_test.parquet",
        "y_train": path / "y_train.parquet",
        "y_val": path / "y_val.parquet",
        "y_test": path / "y_test.parquet",
    }

    return data_paths

def persist_feature_snapshot(config: TabularFeaturesConfig, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame, snapshot_id: str):
    path = Path(f"{config.feature_store_path}/{snapshot_id}")
    path.mkdir(parents=True, exist_ok=True)

    # Expandable for future storage formats
    FREEZE_FORMAT_REGISTRY = {
        "parquet": freeze_parquet,
    }

    freeze_func = FREEZE_FORMAT_REGISTRY[config.storage.format]
    data_paths = freeze_func(
        path, 
        X_train=X_train, 
        X_val=X_val, 
        X_test=X_test, 
        y_train=y_train, 
        y_val=y_val, 
        y_test=y_test, 
        compression=config.storage.compression
    )

    return path, data_paths

def save_input_schema(path: Path, X_train: pd.DataFrame):
    # Stop if raw schema already exists
    schema_path = path / "input_schema.csv"
    if schema_path.exists():
        logger.info(f"Input schema already exists at {schema_path}, skipping save.")
        return

    schema = pd.DataFrame({
        "feature": X_train.columns if isinstance(X_train, pd.DataFrame) else [X_train.name],
        "dtype": X_train.dtypes.astype(str) if isinstance(X_train, pd.DataFrame) else str(X_train.dtype),
        "role": "input",
    })

    schema.to_csv(schema_path, index=False)
    logger.info(f"Input schema saved to {schema_path}")

def save_derived_schema(path: Path, X_train: pd.DataFrame, operator_names: list[str], mode: str):
    # Stop if derived schema already exists
    schema_path = path / "derived_schema.csv"
    if schema_path.exists():
        logger.info(f"Derived schema already exists at {schema_path}, skipping save.")
        return

    operators = [FEATURE_OPERATORS[name]() for name in operator_names]

    X_sample = X_train.head(100)  # small sample to detect dtypes
    derived_features = []
    for op in operators:
        X_sample = op.transform(X_sample)
        for f in op.output_features:
            derived_features.append({
                "feature": f,
                "dtype": str(X_sample[f].dtype),
                "role": "derived",
                "source_operator": op.__class__.__name__,
                "materialized": mode == "materialized",
            })

    derived_schema = pd.DataFrame(derived_features)
    derived_schema.to_csv(schema_path, index=False)
    logger.info(f"Derived schema saved to {schema_path}")

def create_metadata(*, timestamp: str, snapshot_path: Path, schema_path: Path, loader_validation_hash: str, in_memory_hashes: dict, file_hashes: dict, snapshot_identity_hash: str, operators_hash: str, config_hash: str, feature_schema_hash: str, runtime: dict, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame, task: str, duration: float) -> dict:

    metadata = {
        "created_by": "freeze.py",
        "created_at": timestamp,
        "feature_type": "tabular",
        "snapshot_path": str(snapshot_path),
        "snapshot_id": snapshot_path.name,
        "schema_path": str(schema_path),
        "loader_validation_hash": loader_validation_hash,
        "in_memory_hashes": in_memory_hashes,
        "file_hashes": file_hashes,
        "snapshot_identity_hash": snapshot_identity_hash,
        "operators_hash": operators_hash,
        "config_hash": config_hash,
        "feature_schema_hash": feature_schema_hash,
        "runtime": runtime,
        "row_counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "column_count": X_train.shape[1],
        "duration_seconds": duration,
    }

    if task == "classification":
        y_train_series = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train
        y_val_series = y_val.iloc[:, 0] if isinstance(y_val, pd.DataFrame) else y_val
        y_test_series = y_test.iloc[:, 0] if isinstance(y_test, pd.DataFrame) else y_test

        metadata["class_counts"] = {
            "train": y_train_series.value_counts().to_dict(),
            "val": y_val_series.value_counts().to_dict(),
            "test": y_test_series.value_counts().to_dict(),
        }

    return metadata