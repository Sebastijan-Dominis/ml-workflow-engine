"""Unit tests for promotion validation helpers."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from ml.exceptions import ConfigError, UserError
from ml.metadata.schemas.runners.explainability import ExplainabilityArtifacts
from ml.promotion.config.promotion_thresholds import PromotionThresholds
from ml.promotion.constants.constants import RunnersMetadata
from ml.promotion.validation.artifacts import (
    validate_artifacts_consistency,
    validate_explainability_artifacts,
    validate_optional_artifact,
)
from ml.promotion.validation.promotion_thresholds import validate_promotion_thresholds
from ml.promotion.validation.runners import validate_run_dirs, validate_run_ids

pytestmark = pytest.mark.unit


def _args() -> argparse.Namespace:
    """Build minimal CLI args namespace used in validation error messages."""
    return argparse.Namespace(train_run_id="train-1", eval_run_id="eval-1", explain_run_id="exp-1")


def _runners_metadata(
    *,
    train_status: str = "success",
    eval_status: str = "success",
    explain_status: str = "success",
    train_train_run_id: str = "train-1",
    eval_train_run_id: str = "train-1",
    explain_train_run_id: str = "train-1",
    model_path: str | None = None,
    model_hash: str | None = None,
    pipeline_path: str | None = None,
    pipeline_hash: str | None = None,
) -> RunnersMetadata:
    """Construct minimal runners metadata object for validation tests."""

    return cast(
        RunnersMetadata,
        SimpleNamespace(
            training_metadata=SimpleNamespace(
                run_identity=SimpleNamespace(
                    status=train_status,
                    train_run_id=train_train_run_id,
                ),
                artifacts=SimpleNamespace(
                    model_path=model_path,
                    model_hash=model_hash,
                    pipeline_path=pipeline_path,
                    pipeline_hash=pipeline_hash,
                ),
            ),
            evaluation_metadata=SimpleNamespace(
                run_identity=SimpleNamespace(
                    status=eval_status,
                    train_run_id=eval_train_run_id,
                ),
                artifacts=SimpleNamespace(
                    model_path=model_path,
                    model_hash=model_hash,
                    pipeline_path=pipeline_path,
                    pipeline_hash=pipeline_hash,
                ),
            ),
            explainability_metadata=SimpleNamespace(
                run_identity=SimpleNamespace(
                    status=explain_status,
                    train_run_id=explain_train_run_id,
                ),
                artifacts=SimpleNamespace(
                    model_path=model_path,
                    model_hash=model_hash,
                    pipeline_path=pipeline_path,
                    pipeline_hash=pipeline_hash,
                ),
            ),
        ),
    )


def test_validate_run_dirs_passes_when_all_directories_exist(tmp_path: Path) -> None:
    """Accept run-dir inputs when train/eval/explain directories all exist."""
    train_run_dir = tmp_path / "train"
    eval_run_dir = tmp_path / "eval"
    explain_run_dir = tmp_path / "explain"
    train_run_dir.mkdir()
    eval_run_dir.mkdir()
    explain_run_dir.mkdir()

    validate_run_dirs(train_run_dir, eval_run_dir, explain_run_dir)


@pytest.mark.parametrize(
    ("train_exists", "eval_exists", "explain_exists", "expected_message"),
    [
        (False, True, True, "Train run directory does not exist"),
        (True, False, True, "Eval run directory does not exist"),
        (True, True, False, "Explain run directory does not exist"),
    ],
)
def test_validate_run_dirs_raises_for_missing_directory(
    tmp_path: Path,
    train_exists: bool,
    eval_exists: bool,
    explain_exists: bool,
    expected_message: str,
) -> None:
    """Raise UserError naming the first missing required run directory."""
    train_run_dir = tmp_path / "train"
    eval_run_dir = tmp_path / "eval"
    explain_run_dir = tmp_path / "explain"

    if train_exists:
        train_run_dir.mkdir()
    if eval_exists:
        eval_run_dir.mkdir()
    if explain_exists:
        explain_run_dir.mkdir()

    with pytest.raises(UserError, match=expected_message):
        validate_run_dirs(train_run_dir, eval_run_dir, explain_run_dir)


def test_validate_run_ids_passes_when_eval_and_explain_link_to_same_train_run() -> None:
    """Accept linked runs when both eval and explain metadata point to train run."""
    validate_run_ids(args=_args(), runners_metadata=_runners_metadata())


def test_validate_run_ids_raises_when_eval_not_linked_to_train() -> None:
    """Raise UserError when evaluation metadata references a different train run."""
    metadata = _runners_metadata(eval_train_run_id="other-train")

    with pytest.raises(UserError, match="Evaluation run eval-1 is not linked to train run train-1"):
        validate_run_ids(args=_args(), runners_metadata=metadata)


def test_validate_run_ids_raises_when_explain_not_linked_to_train() -> None:
    """Raise UserError when explainability metadata references a different train run."""
    metadata = _runners_metadata(explain_train_run_id="other-train")

    with pytest.raises(UserError, match="Explain run exp-1 is not linked to train run train-1"):
        validate_run_ids(args=_args(), runners_metadata=metadata)


def _write_file(path: Path, content: str = "data") -> str:
    """Create file and return deterministic hash via hash_artifact."""
    path.write_text(content)
    from ml.promotion.validation.artifacts import hash_artifact

    return hash_artifact(path)


def test_validate_artifacts_consistency_passes_on_fully_matching_artifacts(
    tmp_path: Path,
) -> None:
    """Accept when all runs succeeded and model/pipeline artifacts match exactly."""
    model_file = tmp_path / "model.bin"
    pipeline_file = tmp_path / "pipeline.pkl"

    model_hash = _write_file(model_file, "model")
    pipeline_hash = _write_file(pipeline_file, "pipeline")

    metadata = _runners_metadata(
        model_path=str(model_file),
        model_hash=model_hash,
        pipeline_path=str(pipeline_file),
        pipeline_hash=pipeline_hash,
    )

    validate_artifacts_consistency(metadata)


@pytest.mark.parametrize("status_field", ["train", "eval", "explain"])
def test_validate_artifacts_consistency_raises_on_non_success_status(
    tmp_path: Path,
    status_field: str,
) -> None:
    """Raise UserError when any run did not complete successfully."""
    model_file = tmp_path / "model.bin"
    model_hash = _write_file(model_file)

    kwargs = {
        "train_status": "success",
        "eval_status": "success",
        "explain_status": "success",
        "model_path": str(model_file),
        "model_hash": model_hash,
    }
    kwargs[f"{status_field}_status"] = "failed"

    metadata = _runners_metadata(**kwargs)

    with pytest.raises(UserError, match="did not complete successfully"):
        validate_artifacts_consistency(metadata)


def test_validate_artifacts_consistency_raises_on_model_hash_mismatch(
    tmp_path: Path,
) -> None:
    """Raise when runtime model hash differs from metadata hash."""
    model_file = tmp_path / "model.bin"
    _write_file(model_file, "real")

    metadata = _runners_metadata(
        model_path=str(model_file),
        model_hash="different-hash",
    )

    with pytest.raises(UserError, match="Model hash mismatch"):
        validate_artifacts_consistency(metadata)


def test_validate_artifacts_consistency_raises_on_missing_model_path() -> None:
    """Raise when model path is missing."""
    metadata = _runners_metadata(model_path=None, model_hash="hash")

    with pytest.raises(UserError, match="Model path is missing"):
        validate_artifacts_consistency(metadata)


def test_validate_artifacts_consistency_raises_on_missing_model_hash(
    tmp_path: Path,
) -> None:
    """Raise when model hash is missing."""
    model_file = tmp_path / "model.bin"
    _write_file(model_file)

    metadata = _runners_metadata(
        model_path=str(model_file),
        model_hash=None,
    )

    with pytest.raises(UserError, match="Missing model hash"):
        validate_artifacts_consistency(metadata)


def test_validate_artifacts_consistency_raises_on_pipeline_presence_mismatch(
    tmp_path: Path,
) -> None:
    """Raise when pipeline path presence differs across runs."""
    model_file = tmp_path / "model.bin"
    model_hash = _write_file(model_file)

    metadata = _runners_metadata(
        model_path=str(model_file),
        model_hash=model_hash,
    )

    metadata.evaluation_metadata.artifacts.pipeline_hash = "some-hash"

    with pytest.raises(UserError, match="Inconsistent pipeline_hash presence"):
        validate_artifacts_consistency(metadata)


# ---------------------------------------------------------------------------
# validate_optional_artifact
# ---------------------------------------------------------------------------


def test_validate_optional_artifact_passes_when_absent() -> None:
    """Accept when both path and hash are None."""
    validate_optional_artifact(None, None, "artifact")


def test_validate_optional_artifact_raises_on_presence_mismatch() -> None:
    """Raise when path exists but hash does not (or vice versa)."""
    with pytest.raises(UserError, match="Inconsistent presence"):
        validate_optional_artifact("some/path", None, "artifact")


def test_validate_optional_artifact_raises_on_missing_file(
    tmp_path: Path,
) -> None:
    """Raise when artifact path does not exist."""
    fake = tmp_path / "missing.bin"

    with pytest.raises(UserError, match="does not exist"):
        validate_optional_artifact(str(fake), "hash", "artifact")


def test_validate_optional_artifact_raises_on_hash_mismatch(
    tmp_path: Path,
) -> None:
    """Raise when computed hash differs from expected hash."""
    file = tmp_path / "file.bin"
    _write_file(file, "real")

    with pytest.raises(UserError, match="Artifact hash mismatch"):
        validate_optional_artifact(str(file), "wrong-hash", "artifact")


# ---------------------------------------------------------------------------
# validate_explainability_artifacts
# ---------------------------------------------------------------------------


def test_validate_explainability_artifacts_passes(
    tmp_path: Path,
) -> None:
    """Accept when explainability artifacts are consistent."""
    file1 = tmp_path / "f1.bin"
    file2 = tmp_path / "f2.bin"

    hash1 = _write_file(file1, "a")
    hash2 = _write_file(file2, "b")

    artifacts = cast(
        ExplainabilityArtifacts,
        SimpleNamespace(
            top_k_feature_importances_path=str(file1),
            top_k_feature_importances_hash=hash1,
            top_k_shap_importances_path=str(file2),
            top_k_shap_importances_hash=hash2,
        )
    )

    validate_explainability_artifacts(artifacts)


def test_validate_explainability_artifacts_raises_on_mismatch(
    tmp_path: Path,
) -> None:
    """Raise when any explainability artifact is invalid."""
    file1 = tmp_path / "f1.bin"
    _write_file(file1, "a")

    artifacts = cast(
        ExplainabilityArtifacts,
        SimpleNamespace(
            top_k_feature_importances_path=str(file1),
            top_k_feature_importances_hash="wrong",
            top_k_shap_importances_path=None,
            top_k_shap_importances_hash=None,
        )
    )

    with pytest.raises(UserError):
        validate_explainability_artifacts(artifacts)


def test_validate_promotion_thresholds_returns_validated_schema() -> None:
    """Return validated PromotionThresholds instance for a valid threshold payload."""
    payload = {
        "promotion_metrics": {
            "sets": ["val"],
            "metrics": ["f1"],
            "directions": {"f1": "maximize"},
        },
        "thresholds": {"val": {"f1": 0.7}},
        "lineage": {"created_by": "tests", "created_at": "2026-03-05T00:00:00"},
    }

    result = validate_promotion_thresholds(payload)

    assert isinstance(result, PromotionThresholds)
    assert result.thresholds.val["f1"] == pytest.approx(0.7)


def test_validate_promotion_thresholds_wraps_schema_errors_as_config_error() -> None:
    """Raise ConfigError with contextual message when threshold payload is invalid."""
    payload = {
        "promotion_metrics": {
            "sets": ["val"],
            "metrics": ["f1"],
            "directions": {"f1": "maximize"},
        },
        "thresholds": {"val": {"f1": 0.7}},
        "lineage": {"created_by": "tests", "created_at": "not-a-datetime"},
    }

    with pytest.raises(ConfigError, match="Invalid promotion thresholds configuration"):
        validate_promotion_thresholds(payload)
