"""A module for preparing metadata to be stored alongside predictions after inference execution."""
import argparse
from datetime import datetime
from typing import Literal

from ml.modeling.models.feature_lineage import FeatureLineage
from ml.promotion.config.registry_entry import RegistryEntry


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
    """Prepare metadata dictionary to be stored with predictions for monitoring and lineage.

    Args:
        model_metadata: Metadata for the model used in inference.
        args: Command-line arguments.
        run_id: Unique identifier for the inference run.
        timestamp: Current timestamp for partitioning and metadata.
        stage: "production" or "staging" - used for labeling predictions and monitoring.
        cols: List of column names in the predictions.
        snapshot_bindings_id: ID for the snapshot bindings.
        feature_lineage: List of feature lineage objects.
        artifact_type: Type of the artifact (pipeline or model).
        artifact_hash: Hash of the artifact.
        inference_latency_seconds: Latency of the inference process in seconds.

    Returns:
        Dictionary of metadata to be stored with predictions.
    """

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
