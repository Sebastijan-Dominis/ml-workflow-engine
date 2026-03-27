"""A module for storing predictions and related metadata after inference execution in the post-promotion pipeline."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.exceptions import PipelineContractError
from ml.post_promotion.inference.classes.function_returns import PredictionStoringReturn
from ml.post_promotion.inference.classes.predictions_schema import SCHEMA_VERSION
from ml.post_promotion.inference.validation.validate_columns import validate_columns
from ml.promotion.config.registry_entry import RegistryEntry

logger = logging.getLogger(__name__)

def store_predictions(
    *,
    features: pd.DataFrame,
    entity_key: str,
    run_id: str,
    input_hash: pd.Series,
    path: Path,
    timestamp: datetime,
    predictions: pd.Series,
    probabilities: pd.DataFrame,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"]
) -> PredictionStoringReturn:
    """Store predictions along with metadata and hashes for monitoring purposes.

    Args:
        features: Input features used for prediction, required for entity key and monitoring.
        entity_key: Column name in features that serves as the entity identifier for monitoring joins.
        run_id: Unique identifier for the inference run.
        input_hash: Hashes of input rows for monitoring data drift and alignment.
        path: Directory where predictions will be stored.
        timestamp: Current timestamp for partitioning and metadata.
        predictions: Series of prediction results.
        probabilities: DataFrame of prediction probabilities.
        model_metadata: Metadata for the model used for prediction.
        stage: "production" or "staging" - used for labeling predictions and monitoring.

    Returns:
        PredictionStoringReturn: The result of the prediction storing operation.
    """
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / "predictions.parquet"

    # Build schema-safe DataFrame
    df = pd.DataFrame()

    # --- identifiers ---
    df["run_id"] = run_id
    df["timestamp"] = timestamp.isoformat()
    df["prediction_id"] = [uuid4().hex for _ in range(len(features))]

    # --- model metadata ---
    df["model_stage"] = stage
    df["model_version"] = model_metadata.model_version

    # --- entity key (assumes exists) ---
    if entity_key not in features.columns:
        msg = f"Entity key '{entity_key}' not found in features. Cannot store predictions without entity identifier for monitoring joins."
        logger.error(msg)
        raise PipelineContractError(msg)
    df["entity_id"] = features[entity_key]

    # --- hash for alignment ---
    df["input_hash"] = input_hash

    # --- predictions ---
    df["prediction"] = predictions.values

    if not probabilities.empty:
        for i, col in enumerate(probabilities.columns):
            df[f"proba_{i}"] = probabilities[col].astype(float)

    df["schema_version"] = SCHEMA_VERSION

    cols = validate_columns(df)

    tmp_path = file_path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pandas(df), tmp_path)
    tmp_path.rename(file_path)

    logger.info(f"Stored predictions at {file_path}")

    return PredictionStoringReturn(
        file_path=file_path,
        cols = cols
    )
