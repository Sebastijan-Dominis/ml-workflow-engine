"""Tests for persistence helpers used by training scripts.

Includes tests that verify model and metadata files are written to disk
and that the global `models.yaml` registry is updated correctly.
"""

import json
import pytest

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from pathlib import Path

from ml.training.train_scripts.persistence import save_model_pipeline_and_metadata
from ml.training.train_scripts.persistence.update_general_config import (
    update_general_config
)

def test_save_pipeline_and_metadata_writes_files(tmp_path: Path, minimal_training_cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Save a pipeline and metadata, asserting the expected files exist.

    The test redirects the module-level output directories to a temporary
    path and confirms the produced joblib and JSON files contain the
    expected content.
    """

    # Arrange - redirect save paths to tmp
    monkeypatch.setattr(
        save_model_pipeline_and_metadata, "model_dir", tmp_path / "models"
    )
    monkeypatch.setattr(
        save_model_pipeline_and_metadata, "metadata_dir", tmp_path / "metadata"
    )

    pipeline = Pipeline([("model", DummyClassifier())])

    save_model_pipeline_and_metadata.save_pipeline_and_metadata(
        pipeline, minimal_training_cfg
    )

    model_file = tmp_path / "models" / "test_model_v0.joblib"
    metadata_file = tmp_path / "metadata" / "test_model_v0.json"

    assert model_file.exists()
    assert metadata_file.exists()

    data = json.loads(metadata_file.read_text())
    assert data["model_name"] == f"{minimal_training_cfg['name']}_{minimal_training_cfg['version']}"
    assert data["features_version"] == minimal_training_cfg["data"]["features_version"]


def test_update_general_config_creates_entry(tmp_path: Path, minimal_training_cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure `update_general_config` writes an entry into `configs/models.yaml`.

    The test runs in a temporary working directory and asserts the
    models registry file contains the expected entry for the test model.
    """

    monkeypatch.chdir(tmp_path)

    update_general_config(minimal_training_cfg)

    config_path = tmp_path / "configs" / "models.yaml"
    assert config_path.exists()

    import yaml
    with open(config_path) as f:
        data = yaml.safe_load(f)

    key = "test_model_v0"
    assert key in data
    assert data[key]["task"] == "binary_classification"