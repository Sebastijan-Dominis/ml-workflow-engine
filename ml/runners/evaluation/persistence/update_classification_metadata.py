"""Persistence helpers to update classification metadata.

This module provides small, focused functions to open an existing JSON
metadata file, merge evaluation metrics into it, and persist the result. It
is intended to be used by evaluation runners to record per-split metrics and
the selected decision threshold.
"""

# General imports
import json
import logging
from pathlib import Path

# Utility imports
from ml.runners.evaluation.utils.utils import assert_keys

logger = logging.getLogger(__name__)

def get_file(model_configs: dict) -> Path:
    """Return the `Path` to the metadata file declared in the config.

    Args:
        model_configs (dict): Model configuration mapping.

    Returns:
        pathlib.Path: Path to the JSON metadata file.
    """

    metadata_file = Path(model_configs["artifacts"]["metadata"])
    return metadata_file

def load_metadata(metadata_file: Path) -> dict:
    """Load and return JSON metadata from disk.

    Args:
        metadata_file (pathlib.Path): Path to the metadata JSON file.

    Returns:
        dict: Parsed JSON metadata.
    """

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata

def update_content(metadata: dict, evaluation_results: dict, best_threshold: float) -> None:
    """Merge evaluation results and threshold into the metadata dict.

    The function mutates the provided `metadata` mapping by ensuring a
    top-level `metrics` key exists and storing per-split metric dicts.

    Args:
        metadata (dict): Existing metadata mapping to update in-place.
        evaluation_results (dict): Mapping from split name to metrics.
        best_threshold (float): Selected probability threshold.
    """

    metadata.setdefault("metrics", {})
    for split_name, metrics in evaluation_results.items():
        if split_name in metadata["metrics"]:
            logger.warning(f"Overwriting existing metrics for split '{split_name}'.")
        metadata["metrics"][split_name] = metrics
    metadata["threshold"] = best_threshold

def save_metadata(metadata: dict, metadata_file: Path) -> None:
    """Persist metadata mapping to the provided file path as JSON.

    Args:
        metadata (dict): Metadata to write.
        metadata_file (pathlib.Path): File path where JSON will be written.
    """

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

def update_classification_metadata(model_configs: dict, evaluation_results: dict, best_threshold: float) -> None:
    """Top-level helper to update classification metadata on disk.

    Args:
        model_configs (dict): Model configuration mapping containing the
            `artifacts.metadata` path.
        evaluation_results (dict): Mapping of split name to metric dicts.
        best_threshold (float): Selected probability threshold.
    """

    # Step 1 - Ensure required keys are present
    assert_keys(model_configs["artifacts"], ["metadata"])

    # Step 2 - get metadata file path
    metadata_file = get_file(model_configs)

    # Step 3 - load existing metadata
    metadata = load_metadata(metadata_file)

    # Step 4 - update metadata content
    update_content(metadata, evaluation_results, best_threshold)

    # Step 5 - save updated metadata
    save_metadata(metadata, metadata_file)

    # Step 6 - log success message
    logger.info(f"Metadata updated successfully for model '{model_configs['name']}_{model_configs['version']}'.")