"""CLI for running model inference (production + staging) with monitoring-ready outputs."""

import argparse
import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ml.cli.error_handling import resolve_exit_code
from ml.exceptions import InferenceError, PipelineContractError, RuntimeMLError
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.persistence.save_metadata import save_metadata
from ml.logging_config import setup_logging
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.post_promotion.shared.loading.features import prepare_features
from ml.post_promotion.shared.loading.model_registry import get_model_registry_info
from ml.promotion.config.registry_entry import RegistryEntry
from ml.runners.shared.loading.pipeline import load_model_or_pipeline
from ml.utils.hashing.service import hash_artifact
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "V1"

BASE_EXPECTED_COLUMNS = [
    "run_id",
    "prediction_id",
    "timestamp",
    "model_stage",
    "model_version",
    "entity_id",
    "input_hash",
    "prediction",
    "schema_version"
]

PROBA_PREFIX = "proba_"

def validate_columns(df: pd.DataFrame) -> list[str]:
    cols = set(df.columns)

    # Check required base columns
    missing = set(BASE_EXPECTED_COLUMNS) - cols
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check probability columns (optional but must follow pattern)
    proba_cols = [c for c in cols if c.startswith(PROBA_PREFIX)]

    if proba_cols:
        # Ensure consistent indexing (proba_0, proba_1, ...)
        expected = {f"{PROBA_PREFIX}{i}" for i in range(len(proba_cols))}
        if set(proba_cols) != expected:
            raise ValueError(f"Probability columns malformed: {proba_cols}")

    return list(cols)

@dataclass
class ArtifactLoadingReturn:
    artifact: Any
    artifact_hash: str
    artifact_type: Literal["pipeline", "model"]

def load_and_validate_artifact(model_metadata: RegistryEntry) -> ArtifactLoadingReturn:
    pipeline_path = Path(model_metadata.artifacts.pipeline_path or "")
    model_path = Path(model_metadata.artifacts.model_path)
    expected_hash = model_metadata.artifacts.pipeline_hash

    if pipeline_path.exists():
        artifact = load_model_or_pipeline(pipeline_path, target_type="pipeline")
        actual_hash = hash_artifact(pipeline_path)
    elif model_path.exists():
        artifact = load_model_or_pipeline(model_path, target_type="model")
        actual_hash = hash_artifact(model_path)
    else:
        raise PipelineContractError("No valid artifact found.")

    if actual_hash != expected_hash:
        raise PipelineContractError(f"Hash mismatch! Expected {expected_hash}, got {actual_hash}")

    return ArtifactLoadingReturn(
        artifact=artifact,
        artifact_hash=actual_hash,
        artifact_type="pipeline" if pipeline_path.exists() else "model"
    )


# =========================================================
# Prediction
# =========================================================
def predict(X: pd.DataFrame, artifact: Any) -> tuple[pd.Series, pd.DataFrame]:
    try:
        preds = pd.Series(artifact.predict(X), index=X.index)

        if hasattr(artifact, "predict_proba"):
            proba = pd.DataFrame(artifact.predict_proba(X), index=X.index)
        else:
            proba = pd.DataFrame()
    except Exception as e:
        msg = "Error during prediction. "
        logger.exception(msg)
        raise InferenceError(msg) from e

    return preds, proba


# =========================================================
# Stable hashing
# =========================================================
def hash_input_row(row: pd.Series) -> str:
    row = row.sort_index()

    normalized = []
    for v in row.values:
        if pd.isna(v):
            normalized.append("NULL")
        elif isinstance(v, float):
            normalized.append(f"{v:.10g}")  # stable float format
        else:
            normalized.append(str(v))

    row_str = "|".join(normalized)
    return hashlib.sha256(row_str.encode()).hexdigest()


@dataclass
class PredictionStoringReturn:
    file_path: Path
    cols: list[str]

# =========================================================
# Storage (append-only, partitioned)
# =========================================================
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

class InferenceMetadata(BaseModel):
    problem_type: str
    segment: str
    model_version: str
    model_stage: str
    run_id: str
    timestamp: str
    columns: list[str]
    snapshot_bindings_id: str
    feature_lineage: list[FeatureLineage]
    artifact_type: Literal["pipeline", "model"]
    artifact_hash: str
    inference_latency_seconds: float

def validate_inference_metadata(metadata: dict) -> InferenceMetadata:
    """
    Validate the inference metadata against the InferenceMetadata schema.

    Args:
        metadata (dict): The metadata dictionary to validate.

    Returns:
        InferenceMetadata: The validated metadata object.
    """
    try:
        validated_metadata = InferenceMetadata.model_validate(metadata)
        logger.debug("Successfully validated inference metadata.")
        return validated_metadata
    except Exception as e:
        msg = "Error validating inference metadata."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e


def prepare_metadata(
    *,
    model_metadata: RegistryEntry,
    args: argparse.Namespace,
    run_id: str,
    timestamp: datetime,
    stage: Literal["production", "staging"],
    cols: list[str],
    snapshot_bindings_id: str,
    feature_lineage: list[FeatureLineage],
    artifact_type: Literal["pipeline", "model"],
    artifact_hash: str,
    inference_latency_seconds: float
) -> dict:
    metadata = {
        "problem_type": args.problem,
        "segment": args.segment,
        "model_version": model_metadata.model_version,
        "model_stage": stage,
        "run_id": run_id,
        "timestamp": timestamp.isoformat(),
        "columns": cols,
        "snapshot_bindings_id": snapshot_bindings_id,
        "feature_lineage": [f.model_dump() for f in feature_lineage],
        "artifact_type": artifact_type,
        "artifact_hash": artifact_hash,
        "inference_latency_seconds": inference_latency_seconds

    }
    return metadata

# =========================================================
# Inference execution
# =========================================================
def execute_inference(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"],
    timestamp: datetime,
    path: Path,
    run_id: str
):
    """Run inference for a given model and store predictions with monitoring-ready outputs.

    Args:
        args: Command-line arguments.
        model_metadata: Metadata for the model to run inference with.
        stage: "production" or "staging" - used for labeling predictions and monitoring.
        timestamp: Current timestamp for partitioning and metadata.
        path: Directory where predictions will be stored.
    """

    logger.info(f"Running inference for stage={stage} with model version={model_metadata.model_version}...")

    prepare_features_return = prepare_features(
        args=args,
        model_metadata=model_metadata,
        snapshot_bindings_id=args.snapshot_bindings_id
    )

    features = prepare_features_return.features
    entity_key = prepare_features_return.entity_key
    feature_lineage = prepare_features_return.feature_lineage

    artifact_loading_return = load_and_validate_artifact(model_metadata)

    artifact = artifact_loading_return.artifact
    artifact_hash = artifact_loading_return.artifact_hash
    artifact_type = artifact_loading_return.artifact_type

    features_for_prediction = features.drop(columns=[entity_key], errors="ignore")
    features_for_prediction = features_for_prediction.sort_index(axis=1)

    input_hash = features_for_prediction.apply(hash_input_row, axis=1)

    start_time = time.perf_counter()

    preds, proba = predict(features_for_prediction, artifact)

    end_time = time.perf_counter()

    duration = end_time - start_time

    logger.info(f"Inference completed in {duration:.4f} seconds. Storing predictions...")

    prediction_return = store_predictions(
        features=features,
        entity_key=entity_key,
        run_id=run_id,
        timestamp=timestamp,
        path=path,
        predictions=preds,
        probabilities=proba,
        model_metadata=model_metadata,
        stage=stage,
        input_hash=input_hash
    )

    cols = prediction_return.cols

    metadata_raw = prepare_metadata(
        model_metadata=model_metadata,
        args=args,
        run_id=run_id,
        timestamp=timestamp,
        stage=stage,
        snapshot_bindings_id=args.snapshot_bindings_id,
        feature_lineage=feature_lineage,
        artifact_hash=artifact_hash,
        artifact_type=artifact_type,
        inference_latency_seconds=duration,
        cols=cols
    )

    metadata = validate_inference_metadata(metadata_raw)

    save_metadata(
        metadata = metadata.model_dump(exclude_none=True),
        target_dir=path
    )

# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for production and staging models with monitoring-ready outputs.")

    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Model problem, e.g., 'no_show'"
    )

    parser.add_argument(
        "--segment",
        type=str,
        required=True,
        help="Model segment name, e.g., 'city_hotel_online_ta'"
    )

    parser.add_argument(
        "--snapshot-bindings-id",
        required=True,
        help = "A snapshot binding to define which snapshot to load for each feature set."
    )

    parser.add_argument(
        "--logging-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main() -> int:
    args = parse_args()

    timestamp = datetime.now()
    run_id = f"{iso_no_colon(timestamp)}_{uuid4().hex[:8]}"
    base_path = Path("predictions") / args.problem / args.segment

    run_dir = base_path / run_id

    log_path = run_dir / "inference.log"

    setup_logging(log_path, getattr(logging, args.logging_level, logging.INFO))

    try:
        model_registry_info = get_model_registry_info(args)

        prod_meta = model_registry_info.prod_meta
        stage_meta = model_registry_info.stage_meta

        if prod_meta is not None:
            execute_inference(
                args=args,
                model_metadata=prod_meta,
                stage="production",
                timestamp=timestamp,
                path=run_dir / "production",
                run_id=run_id
            )

        if stage_meta is not None:
            execute_inference(
                args=args,
                model_metadata=stage_meta,
                stage="staging",
                timestamp=timestamp,
                path=run_dir / "staging",
                run_id=run_id
            )

        return 0

    except Exception as e:
        logger.exception("Inference failed")
        return resolve_exit_code(e)


if __name__ == "__main__":
    sys.exit(main())
