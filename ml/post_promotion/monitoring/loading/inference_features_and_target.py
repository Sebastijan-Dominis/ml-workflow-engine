"""Module for loading inference features and target for post-promotion monitoring."""
import argparse
from pathlib import Path
from typing import Literal

from ml.metadata.validation.post_promotion.infer import validate_inference_metadata
from ml.post_promotion.monitoring.classes.function_returns import InferenceFeaturesAndTarget
from ml.post_promotion.shared.loading.features import prepare_features
from ml.promotion.config.registry_entry import RegistryEntry
from ml.utils.loaders import load_json
from ml.utils.snapshots.snapshot_path import get_snapshot_path


def load_inference_features_and_target(
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    stage: Literal["production", "staging"]
) -> InferenceFeaturesAndTarget:
    """Load inference features and target for post-promotion monitoring.

    Args:
        args: Command-line arguments containing necessary identifiers.
        model_metadata: Metadata for the model being evaluated.
        stage: The stage for which to load inference data ('production' or 'staging').

    Returns:
        An InferenceFeaturesAndTarget object containing the features and target.
    """
    inference_dir = Path("predictions") / args.problem / args.segment
    snapshot_path = get_snapshot_path(args.inference_run_id, inference_dir)
    inference_run_dir = snapshot_path / stage

    inference_metadata_path = inference_run_dir / "metadata.json"
    inference_metadata_raw = load_json(inference_metadata_path)
    inference_metadata = validate_inference_metadata(inference_metadata_raw)

    # Assumes supervised inference. Modify as needed for unsupervised tasks.
    inference_features_return = prepare_features(
        args=args,
        model_metadata=model_metadata,
        snapshot_bindings_id=inference_metadata.snapshot_bindings_id
    )
    inference_features = inference_features_return.features

    result = InferenceFeaturesAndTarget(
        features=inference_features,
        target=inference_features_return.target
    )
    return result
