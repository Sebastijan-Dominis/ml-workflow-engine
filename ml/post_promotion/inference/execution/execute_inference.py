import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from ml.io.persistence.save_metadata import save_metadata
from ml.metadata.validation.post_promotion.infer import validate_inference_metadata
from ml.post_promotion.inference.execution.predict import predict
from ml.post_promotion.inference.hashing.input_row import hash_input_row
from ml.post_promotion.inference.loading.artifact import load_and_validate_artifact
from ml.post_promotion.inference.persistence.prepare_metadata import prepare_metadata
from ml.post_promotion.inference.persistence.store_predictions import store_predictions
from ml.post_promotion.shared.loading.features import prepare_features
from ml.promotion.config.registry_entry import RegistryEntry

logger = logging.getLogger(__name__)

def execute_inference(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"],
    timestamp: datetime,
    path: Path,
    run_id: str
) -> None:
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
