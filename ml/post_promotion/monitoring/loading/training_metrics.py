"""Module for loading training metrics for post-promotion monitoring."""
import argparse
from pathlib import Path

from ml.modeling.models.metrics import TrainingMetrics
from ml.modeling.validation.metrics import validate_training_metrics
from ml.promotion.config.registry_entry import RegistryEntry
from ml.utils.loaders import load_json


def load_training_metrics_file(
    args: argparse.Namespace,
    model_metadata: RegistryEntry
) -> TrainingMetrics:
    """Load training metrics for post-promotion monitoring.

    Args:
        args: Command-line arguments containing necessary identifiers.
        model_metadata: Metadata for the model being monitored.

    Returns:
        TrainingMetrics: The loaded and validated training metrics.
    """
    training_metrics_path = Path("experiments") / args.problem / args.segment / model_metadata.model_version / model_metadata.experiment_id / "training" / model_metadata.train_run_id / "metrics.json"
    training_metrics_raw = load_json(training_metrics_path)
    training_metrics = validate_training_metrics(training_metrics_raw)
    return training_metrics
