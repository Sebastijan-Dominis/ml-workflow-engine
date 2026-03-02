"""Persistence and metadata helpers for tabular feature freezing outputs."""

import logging
from pathlib import Path

import pandas as pd

from ml.feature_freezing.freeze_strategies.tabular.config.models import \
    TabularFeaturesConfig
from ml.registry.feature_operators import FEATURE_OPERATORS

logger = logging.getLogger(__name__)

def freeze_parquet(
    path: Path, 
    *, 
    features: pd.DataFrame,
    compression=None
) -> Path:
    """Persist feature dataframe as parquet and return output file path.

    Args:
        path: Snapshot directory path.
        features: Features dataframe to persist.
        compression: Optional parquet compression codec.

    Returns:
        Path: Path to persisted parquet file.
    """
    features.to_parquet(path / "features.parquet", index=False, compression=compression)
    
    logger.info(f"Tabular features saved to {path}")

    data_path = path / "features.parquet"

    return data_path

def persist_feature_snapshot(
        config: TabularFeaturesConfig, 
        *,
        features: pd.DataFrame,
        snapshot_id: str
    ) -> tuple[Path, Path]:
    """Persist frozen feature snapshot and return snapshot/data paths.

    Args:
        config: Tabular feature freezing configuration.
        features: Features dataframe to persist.
        snapshot_id: Snapshot identifier.

    Returns:
        tuple[Path, Path]: Snapshot directory path and persisted data-file path.
    """

    path = Path(f"{config.feature_store_path}/{snapshot_id}")
    path.mkdir(parents=True, exist_ok=True)

    # Expandable for future storage formats
    FREEZE_FORMAT_REGISTRY = {
        "parquet": freeze_parquet,
    }

    freeze_func = FREEZE_FORMAT_REGISTRY[config.storage.format]
    data_path = freeze_func(
        path, 
        features=features,
        compression=config.storage.compression
    )

    return path, data_path

def save_input_schema(path: Path, features: pd.DataFrame):
    """Persist input schema CSV when missing.

    Args:
        path: Snapshot directory path.
        features: Input dataframe for schema extraction.

    Returns:
        None: This function writes schema side effects only.
    """

    # Stop if raw schema already exists
    schema_path = path / "input_schema.csv"
    if schema_path.exists():
        logger.info(f"Input schema already exists at {schema_path}, skipping save.")
        return

    schema = pd.DataFrame({
        "feature": features.columns if isinstance(features, pd.DataFrame) else [features.name],
        "dtype": features.dtypes.astype(str) if isinstance(features, pd.DataFrame) else str(features.dtype),
        "role": "input",
    })

    schema.to_csv(schema_path, index=False)
    logger.info(f"Input schema saved to {schema_path}")

def save_derived_schema(
    path: Path, 
    *,
    features: pd.DataFrame, 
    operator_names: list[str], 
    mode: str
):
    """Persist derived schema CSV inferred from configured operators.

    Args:
        path: Snapshot directory path.
        features: Sample features dataframe.
        operator_names: Ordered operator names used to derive features.
        mode: Operator execution mode.

    Returns:
        None: This function writes schema side effects only.
    """

    # Stop if derived schema already exists
    schema_path = path / "derived_schema.csv"
    if schema_path.exists():
        logger.info(f"Derived schema already exists at {schema_path}, skipping save.")
        return

    operators = [FEATURE_OPERATORS[name]() for name in operator_names]

    X_sample = features.head(100)  # small sample to detect dtypes
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

def create_metadata(
    *, 
    timestamp: str, 
    snapshot_path: Path, 
    schema_path: Path, 
    data_lineage: list[dict], 
    in_memory_hash: str, 
    file_hash: str, 
    operators_hash: str, 
    config_hash: str, 
    feature_schema_hash: str, 
    runtime: dict, 
    features: pd.DataFrame, 
    duration: float,
    owner: str
) -> dict:
    """Build final metadata payload for a frozen tabular feature snapshot.

    Args:
        timestamp: Snapshot creation timestamp.
        snapshot_path: Snapshot directory path.
        schema_path: Input schema file path.
        data_lineage: Data lineage entries backing the features.
        in_memory_hash: Hash of in-memory features frame.
        file_hash: Hash of persisted feature artifact.
        operators_hash: Hash representing applied operators.
        config_hash: Feature-freezing config hash.
        feature_schema_hash: Hash of feature schema representation.
        runtime: Runtime metadata payload.
        features: Persisted features dataframe.
        duration: Snapshot creation duration in seconds.
        owner: Snapshot owner identifier.

    Returns:
        dict: Metadata payload ready for persistence.
    """

    metadata = {
        "created_by": "freeze.py",
        "created_at": timestamp,
        "owner": owner,
        "feature_type": "tabular",
        "snapshot_path": str(snapshot_path),
        "snapshot_id": snapshot_path.name,
        "schema_path": str(schema_path),
        "data_lineage": data_lineage,
        "in_memory_hash": in_memory_hash,
        "file_hash": file_hash,
        "operators_hash": operators_hash,
        "config_hash": config_hash,
        "feature_schema_hash": feature_schema_hash,
        "runtime": runtime,
        "row_count": features.shape[0],
        "column_count": features.shape[1],
        "duration_seconds": duration
    }

    return metadata