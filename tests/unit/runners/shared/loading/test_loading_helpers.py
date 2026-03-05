"""Unit tests for shared runner loading helper utilities."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import joblib
import pytest
from ml.exceptions import DataError, PipelineContractError
from ml.runners.shared.loading.get_snapshot_binding_from_training_metadata import (
    get_snapshot_binding_from_training_metadata,
)
from ml.runners.shared.loading.pipeline import load_model_or_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pytestmark = pytest.mark.unit


class _DummyModel:
    """Simple pickleable dummy model used for AllowedModels type checks."""

    def __init__(self, value: int = 0):
        self.value = value


def test_load_model_or_pipeline_raises_when_file_missing(tmp_path: Path) -> None:
    """Raise PipelineContractError when target artifact file path does not exist."""
    missing = tmp_path / "missing.joblib"

    with pytest.raises(PipelineContractError, match="File not found"):
        load_model_or_pipeline(missing, "pipeline")


def test_load_model_or_pipeline_wraps_joblib_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Wrap deserialization failures from joblib as PipelineContractError."""
    file_path = tmp_path / "artifact.joblib"
    file_path.write_bytes(b"not-used")

    def _failing_load(_file_obj: Any) -> Any:
        raise ValueError("broken file")

    monkeypatch.setattr("ml.runners.shared.loading.pipeline.joblib.load", _failing_load)

    with pytest.raises(PipelineContractError, match="Error loading from"):
        load_model_or_pipeline(file_path, "pipeline")


def test_load_model_or_pipeline_returns_pipeline_when_type_matches(tmp_path: Path) -> None:
    """Return deserialized Pipeline object when loading pipeline artifact."""
    file_path = tmp_path / "pipeline.joblib"
    pipeline = Pipeline([("scaler", StandardScaler())])
    joblib.dump(pipeline, file_path)

    loaded = load_model_or_pipeline(file_path, "pipeline")

    assert isinstance(loaded, Pipeline)
    assert loaded.steps[0][0] == "scaler"


def test_load_model_or_pipeline_raises_when_pipeline_type_is_wrong(tmp_path: Path) -> None:
    """Raise PipelineContractError when pipeline loader receives non-Pipeline artifact."""
    file_path = tmp_path / "not_pipeline.joblib"
    joblib.dump({"not": "pipeline"}, file_path)

    with pytest.raises(PipelineContractError, match="Expected a Pipeline object"):
        load_model_or_pipeline(file_path, "pipeline")


def test_load_model_or_pipeline_returns_model_when_type_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Return deserialized model object when artifact matches AllowedModels runtime type."""

    monkeypatch.setattr("ml.runners.shared.loading.pipeline.AllowedModels", _DummyModel)

    file_path = tmp_path / "model.joblib"
    joblib.dump(_DummyModel(7), file_path)

    loaded = load_model_or_pipeline(file_path, "model")

    assert isinstance(loaded, _DummyModel)
    assert loaded.value == 7


def test_load_model_or_pipeline_raises_when_model_type_is_wrong(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Raise PipelineContractError when model loader receives artifact outside AllowedModels."""

    monkeypatch.setattr("ml.runners.shared.loading.pipeline.AllowedModels", _DummyModel)

    file_path = tmp_path / "bad_model.joblib"
    joblib.dump("not-a-model", file_path)

    with pytest.raises(PipelineContractError, match="Expected a model object"):
        load_model_or_pipeline(file_path, "model")


def test_get_snapshot_binding_from_training_metadata_returns_feature_lineage() -> None:
    """Return feature-lineage list when training metadata contains snapshot binding."""
    lineage = [SimpleNamespace(name="feature_a"), SimpleNamespace(name="feature_b")]
    training_metadata = cast(
        Any,
        SimpleNamespace(lineage=SimpleNamespace(feature_lineage=lineage)),
    )

    snapshot_binding = get_snapshot_binding_from_training_metadata(training_metadata)

    assert snapshot_binding == lineage


def test_get_snapshot_binding_from_training_metadata_raises_when_binding_missing() -> None:
    """Raise DataError when training metadata lineage does not include snapshot binding."""
    training_metadata = cast(
        Any,
        SimpleNamespace(lineage=SimpleNamespace(feature_lineage=None)),
    )

    with pytest.raises(DataError, match="No snapshot binding found"):
        get_snapshot_binding_from_training_metadata(training_metadata)
