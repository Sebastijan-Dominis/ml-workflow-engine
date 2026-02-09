"""Tests for evaluation persistence helpers.

These tests cover reading and updating JSON metadata files used by
the evaluation step for classification models.
"""

import json
from pathlib import Path

import pytest

from ml.runners.evaluation.persistence.update_classification_metadata import (
    get_file,
    load_metadata,
    save_metadata,
    update_classification_metadata,
    update_content,
)


def test_get_file(tmp_path: Path) -> None:
    """Test that get_file correctly resolves metadata file path from config."""

    # Create dummy config
    cfg = {
        "artifacts": {
            "metadata": str(tmp_path / "metadata.json"),
        },
    }

    metadata_file = get_file(cfg)

    assert metadata_file == tmp_path / "metadata.json"

def test_load_metadata(tmp_path: Path) -> None:
    """Test that load_metadata correctly reads JSON metadata from disk."""

    # Create dummy metadata file
    metadata_content = {
        "name": "test_model",
        "version": "v1",
    }
    metadata_file = tmp_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata_content, f)

    # Load the metadata
    loaded_metadata = load_metadata(metadata_file)

    assert loaded_metadata == metadata_content

def test_update_content() -> None:
    """Test that update_content correctly merges evaluation results into metadata."""

    # Initial metadata
    metadata = {
        "name": "test_model",
        "version": "v1",
        "metrics": {
            "train": {"accuracy": 0.8}
        }
    }

    # Evaluation results to merge
    evaluation_results = {
        "train": {"accuracy": 0.9, "f1": 0.8},
        "val": {"accuracy": 0.85, "f1": 0.75},
    }
    best_threshold = 0.6

    # Update the metadata
    update_content(metadata, evaluation_results, best_threshold)

    # Check that metrics and threshold were added
    assert "metrics" in metadata
    assert metadata["metrics"]["train"] == {"accuracy": 0.9, "f1": 0.8}
    assert metadata["metrics"]["val"] == {"accuracy": 0.85, "f1": 0.75}
    assert metadata["threshold"] == best_threshold

def test_save_metadata(tmp_path: Path) -> None:
    """Test that save_metadata correctly writes metadata to disk as JSON."""

    # Metadata to save
    metadata = {
        "name": "test_model",
        "version": "v1",
        "metrics": {
            "train": {"accuracy": 0.9},
            "val": {"accuracy": 0.85},
        },
        "threshold": 0.6,
    }

    metadata_file = tmp_path / "metadata.json"

    # Save the metadata
    save_metadata(metadata, metadata_file)

    # Read back the file and check contents
    with open(metadata_file, "r") as f:
        loaded_metadata = json.load(f)

    assert loaded_metadata == metadata

@pytest.mark.integration
def test_update_classification_metadata(tmp_path: Path) -> None:
    """Test that update_classification_metadata correctly updates metadata on disk."""

    # Initial metadata
    initial_metadata = {
        "name": "test_model",
        "version": "v1",
    }
    metadata_file = tmp_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(initial_metadata, f)

    # Dummy model config
    model_configs = {
        "name": "test_model",
        "version": "v1",
        "artifacts": {
            "metadata": str(metadata_file),
        },
    }

    # Evaluation results to merge
    evaluation_results = {
        "train": {"accuracy": 0.9, "f1": 0.8},
        "val": {"accuracy": 0.85, "f1": 0.75},
    }
    best_threshold = 0.6

    # Update the classification metadata
    update_classification_metadata(model_configs, evaluation_results, best_threshold)

    # Load back the metadata and check contents
    with open(metadata_file, "r") as f:
        updated_metadata = json.load(f)

    assert "metrics" in updated_metadata
    assert updated_metadata["metrics"]["train"] == {"accuracy": 0.9, "f1": 0.8}
    assert updated_metadata["metrics"]["val"] == {"accuracy": 0.85, "f1": 0.75}
    assert updated_metadata["threshold"] == best_threshold