import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from datetime import datetime
import pandas as pd

from ml.registry.feature_operators import FEATURE_OPERATORS
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

def freeze_parquet(path: Path, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame, compression=None):
    X_train.to_parquet(path / "X_train.parquet", index=False, compression=compression)
    X_val.to_parquet(path / "X_val.parquet", index=False, compression=compression)
    X_test.to_parquet(path / "X_test.parquet", index=False, compression=compression)

    y_train.to_parquet(path / "y_train.parquet", index=False, compression=compression)
    y_val.to_parquet(path / "y_val.parquet", index=False, compression=compression)
    y_test.to_parquet(path / "y_test.parquet", index=False, compression=compression)    
    
    logger.info(f"Tabular features saved to {path}")

def persist_feature_snapshot(config: TabularFeaturesConfig, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame, now: str):
    path = Path(f"{config.feature_store_path}/{now}")
    path.mkdir(parents=True, exist_ok=True)

    # Expandable for future storage formats
    FREEZE_FORMAT_REGISTRY = {
        "parquet": freeze_parquet,
    }

    freeze_func = FREEZE_FORMAT_REGISTRY[config.storage.format]
    freeze_func(path, X_train, X_val, X_test, y_train, y_val, y_test, config.storage.compression)

    return path

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

def create_metadata(snapshot_path: Path, schema_path: Path, data_hash: str, train_schema_hash: str, val_schema_hash: str, test_schema_hash: str, operators_hash: str, config_hash: str, feature_set_hash: str, git_commit: str | None, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame, task: str) -> dict:
    metadata = {
        "created_by": "freeze.py",
        "created_at": datetime.now().isoformat(),
        "feature_type": "tabular",

        "snapshot_path": str(snapshot_path),
        "schema_path": str(schema_path),

        "data_hash": data_hash,
        "schema_hashes": {
            "train": train_schema_hash,
            "val": val_schema_hash,
            "test": test_schema_hash,
        },
        "operators_hash": operators_hash,
        "config_hash": config_hash,
        "feature_set_hash": feature_set_hash,
        "git_commit": git_commit,

        "row_counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "column_count": X_train.shape[1],
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