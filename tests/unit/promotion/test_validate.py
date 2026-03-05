"""Unit tests for promotion validation helpers."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from ml.exceptions import ConfigError, UserError
from ml.promotion.config.models import PromotionThresholds
from ml.promotion.constants.constants import RunnersMetadata
from ml.promotion.validation.validate import (
    validate_explainability_artifacts_consistency,
    validate_promotion_thresholds,
    validate_run_dirs,
    validate_run_ids,
)

pytestmark = pytest.mark.unit


def _args() -> argparse.Namespace:
    """Build minimal CLI args namespace used in validation error messages."""
    return argparse.Namespace(train_run_id="train-1", eval_run_id="eval-1", explain_run_id="exp-1")


def _runners_metadata(
    *,
    eval_train_run_id: str = "train-1",
    explain_train_run_id: str = "train-1",
    explain_status: str = "success",
    train_model_hash: str = "model-hash-1",
    explain_model_hash: str = "model-hash-1",
    train_pipeline_hash: str = "pipe-hash-1",
    explain_pipeline_hash: str = "pipe-hash-1",
) -> RunnersMetadata:
    """Construct minimal runners metadata object with configurable mismatch scenarios."""
    return cast(
        RunnersMetadata,
        SimpleNamespace(
            training_metadata=SimpleNamespace(
                run_identity=SimpleNamespace(train_run_id="train-1"),
                artifacts=SimpleNamespace(model_hash=train_model_hash, pipeline_hash=train_pipeline_hash),
            ),
            evaluation_metadata=SimpleNamespace(run_identity=SimpleNamespace(train_run_id=eval_train_run_id)),
            explainability_metadata=SimpleNamespace(
                run_identity=SimpleNamespace(train_run_id=explain_train_run_id, status=explain_status),
                artifacts=SimpleNamespace(model_hash=explain_model_hash, pipeline_hash=explain_pipeline_hash),
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


def test_validate_explainability_artifacts_consistency_passes_on_matching_hashes() -> None:
    """Accept explainability artifacts when status is success and hashes match training artifacts."""
    validate_explainability_artifacts_consistency(_runners_metadata(), _args())


def test_validate_explainability_artifacts_consistency_raises_when_status_not_success() -> None:
    """Raise UserError when explainability run did not complete successfully."""
    metadata = _runners_metadata(explain_status="failed")

    with pytest.raises(UserError, match="did not complete successfully"):
        validate_explainability_artifacts_consistency(metadata, _args())


def test_validate_explainability_artifacts_consistency_raises_when_model_hash_differs() -> None:
    """Raise UserError when explainability model hash does not match training model hash."""
    metadata = _runners_metadata(explain_model_hash="different-model-hash")

    with pytest.raises(UserError, match="Model hash in explain run exp-1 does not match"):
        validate_explainability_artifacts_consistency(metadata, _args())


def test_validate_explainability_artifacts_consistency_raises_when_pipeline_hash_missing() -> None:
    """Raise UserError when explainability artifacts do not include pipeline hash."""
    metadata = _runners_metadata(explain_pipeline_hash="")

    with pytest.raises(UserError, match="is missing pipeline hash artifact"):
        validate_explainability_artifacts_consistency(metadata, _args())


def test_validate_explainability_artifacts_consistency_raises_when_pipeline_hash_differs() -> None:
    """Raise UserError when explainability pipeline hash differs from training pipeline hash."""
    metadata = _runners_metadata(explain_pipeline_hash="different-pipeline-hash")

    with pytest.raises(UserError, match="Pipeline hash in explain run exp-1 does not match"):
        validate_explainability_artifacts_consistency(metadata, _args())


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
