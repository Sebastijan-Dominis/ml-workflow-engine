"""Unit tests for model/pipeline artifact hash integrity validation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from ml.exceptions import PipelineContractError
from ml.modeling.models.artifacts import Artifacts
from ml.runners.shared.logical_config.validate_model_and_pipeline import (
    validate_model_and_pipeline,
)

pytestmark = pytest.mark.unit


def _make_training_metadata(artifacts: Artifacts) -> SimpleNamespace:
    """Create a minimal validated-metadata stub exposing an `artifacts` payload."""
    return SimpleNamespace(artifacts=artifacts)


def test_validate_model_and_pipeline_returns_artifacts_when_hashes_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return artifacts payload unchanged when both model and pipeline hashes match."""
    model_file = tmp_path / "model.joblib"
    pipeline_file = tmp_path / "pipeline.joblib"
    model_file.write_bytes(b"m")
    pipeline_file.write_bytes(b"p")

    artifacts = Artifacts(
        model_hash="model-hash",
        model_path=str(model_file),
        pipeline_path=str(pipeline_file),
        pipeline_hash="pipe-hash",
    )

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.load_json",
        lambda _path: {"metadata": "raw"},
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.validate_training_metadata",
        lambda _raw: _make_training_metadata(artifacts),
    )
    hash_map = {
        str(model_file): "model-hash",
        str(pipeline_file): "pipe-hash",
    }
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.hash_artifact",
        lambda path: hash_map[str(path)],
    )

    result = validate_model_and_pipeline(tmp_path)

    assert result == artifacts


def test_validate_model_and_pipeline_raises_when_model_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `PipelineContractError` when persisted model hash differs from expected hash."""
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"m")

    artifacts = Artifacts(
        model_hash="expected-model-hash",
        model_path=str(model_file),
        pipeline_path=None,
        pipeline_hash=None,
    )

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.load_json",
        lambda _path: {"metadata": "raw"},
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.validate_training_metadata",
        lambda _raw: _make_training_metadata(artifacts),
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.hash_artifact",
        lambda _path: "actual-model-hash",
    )

    with pytest.raises(PipelineContractError, match="Model hash mismatch"):
        validate_model_and_pipeline(tmp_path)


def test_validate_model_and_pipeline_raises_when_pipeline_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `PipelineContractError` when optional pipeline artifact hash mismatches."""
    model_file = tmp_path / "model.joblib"
    pipeline_file = tmp_path / "pipeline.joblib"
    model_file.write_bytes(b"m")
    pipeline_file.write_bytes(b"p")

    artifacts = Artifacts(
        model_hash="model-hash",
        model_path=str(model_file),
        pipeline_path=str(pipeline_file),
        pipeline_hash="expected-pipeline-hash",
    )

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.load_json",
        lambda _path: {"metadata": "raw"},
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.validate_training_metadata",
        lambda _raw: _make_training_metadata(artifacts),
    )
    hash_map = {
        str(model_file): "model-hash",
        str(pipeline_file): "actual-pipeline-hash",
    }
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.hash_artifact",
        lambda path: hash_map[str(path)],
    )

    with pytest.raises(PipelineContractError, match="Pipeline hash mismatch"):
        validate_model_and_pipeline(tmp_path)


def test_validate_model_and_pipeline_skips_pipeline_check_when_pipeline_file_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Skip optional pipeline hash validation when metadata points to a non-existing file."""
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"m")
    missing_pipeline_file = tmp_path / "missing_pipeline.joblib"

    artifacts = Artifacts(
        model_hash="model-hash",
        model_path=str(model_file),
        pipeline_path=str(missing_pipeline_file),
        pipeline_hash="expected-but-unused",
    )

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.load_json",
        lambda _path: {"metadata": "raw"},
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.validate_training_metadata",
        lambda _raw: _make_training_metadata(artifacts),
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_model_and_pipeline.hash_artifact",
        lambda path: "model-hash" if str(path) == str(model_file) else "unexpected",
    )

    result = validate_model_and_pipeline(tmp_path)

    assert result == artifacts
