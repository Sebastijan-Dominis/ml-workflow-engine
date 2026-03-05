"""Unit tests for promotion payload preparation helpers."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.exceptions import UserError
from ml.promotion.constants.constants import PreviousProductionRunIdentity
from ml.promotion.persistence.prepare import prepare_metadata, prepare_run_information

pytestmark = pytest.mark.unit


def _args(*, stage: str) -> argparse.Namespace:
    """Build minimal args namespace consumed by prepare helpers."""
    return argparse.Namespace(
        stage=stage,
        version="v1",
        experiment_id="exp-1",
        train_run_id="train-1",
        eval_run_id="eval-1",
        explain_run_id="explain-1",
    )


def test_prepare_run_information_adds_production_fields_and_core_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build production run-info with promotion identifiers and expected core fields."""
    args = _args(stage="production")
    training_metadata = cast(
        Any,
        SimpleNamespace(
            lineage=SimpleNamespace(
                feature_lineage=[
                    SimpleNamespace(model_dump=lambda: {"name": "feature_a"}),
                    SimpleNamespace(model_dump=lambda: {"name": "feature_b"}),
                ]
            )
        ),
    )
    explainability_metadata = cast(
        Any,
        SimpleNamespace(artifacts=SimpleNamespace(model_dump=lambda: {"model_path": "model.cbm"})),
    )

    monkeypatch.setattr("ml.promotion.persistence.prepare.get_pipeline_cfg_hash", lambda _: "pipeline-hash-1")

    result = prepare_run_information(
        args=args,
        experiment_id="exp-1",
        train_run_id="train-1",
        eval_run_id="eval-1",
        explain_run_id="explain-1",
        run_id="prom-1",
        timestamp="20260305T000000",
        training_metadata=training_metadata,
        explainability_metadata=explainability_metadata,
        metrics={"val": {"f1": 0.81}},
        git_commit="commit-abc",
    )

    assert result["model_version"] == "v1"
    assert result["pipeline_cfg_hash"] == "pipeline-hash-1"
    assert result["feature_lineage"] == [{"name": "feature_a"}, {"name": "feature_b"}]
    assert result["artifacts"] == {"model_path": "model.cbm"}
    assert result["promotion_id"] == "prom-1"
    assert result["promoted_at"] == "20260305T000000"


def test_prepare_run_information_adds_staging_fields_for_staging_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build staging run-info with staging identifiers instead of production keys."""
    args = _args(stage="staging")
    training_metadata = cast(
        Any,
        SimpleNamespace(lineage=SimpleNamespace(feature_lineage=[SimpleNamespace(model_dump=lambda: {"name": "feature_a"})])),
    )
    explainability_metadata = cast(Any, SimpleNamespace(artifacts=SimpleNamespace(model_dump=lambda: {"model_path": "model.cbm"})))

    monkeypatch.setattr("ml.promotion.persistence.prepare.get_pipeline_cfg_hash", lambda _: "pipeline-hash-1")

    result = prepare_run_information(
        args=args,
        experiment_id="exp-1",
        train_run_id="train-1",
        eval_run_id="eval-1",
        explain_run_id="explain-1",
        run_id="stage-1",
        timestamp="20260305T000001",
        training_metadata=training_metadata,
        explainability_metadata=explainability_metadata,
        metrics={"val": {"f1": 0.81}},
        git_commit="commit-abc",
    )

    assert result["staging_id"] == "stage-1"
    assert result["staged_at"] == "20260305T000001"
    assert "promotion_id" not in result
    assert "promoted_at" not in result


def test_prepare_metadata_production_validates_and_includes_beats_previous(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build production metadata, route through production validation, and include beats_previous flag."""
    args = _args(stage="production")
    thresholds = cast(Any, SimpleNamespace(model_dump=lambda: {"thresholds": {"val": {"f1": 0.7}}}))
    prev_identity = PreviousProductionRunIdentity(
        experiment_id="exp-old",
        train_run_id="train-old",
        eval_run_id="eval-old",
        explain_run_id="explain-old",
        promotion_id="prom-old",
    )

    validate_calls: list[str] = []

    monkeypatch.setattr("ml.promotion.persistence.prepare.get_conda_env_export", lambda: "env export")
    monkeypatch.setattr("ml.promotion.persistence.prepare.hash_environment", lambda _: "env-hash-prom")
    monkeypatch.setattr("ml.promotion.persistence.prepare.get_training_conda_env_hash", lambda _: "env-hash-prom")
    monkeypatch.setattr("ml.promotion.persistence.prepare.hash_thresholds", lambda _: "threshold-hash-1")

    def _validate(metadata: dict, stage: str) -> Any:
        validate_calls.append(stage)
        return SimpleNamespace(model_dump=lambda **kwargs: metadata)

    monkeypatch.setattr("ml.promotion.persistence.prepare.validate_promotion_metadata", _validate)

    result = prepare_metadata(
        run_id="prom-1",
        args=args,
        metrics={"val": {"f1": 0.82}},
        previous_production_metrics={"val": {"f1": 0.79}},
        promotion_thresholds=thresholds,
        promoted=True,
        beats_previous=True,
        reason="all good",
        git_commit="commit-abc",
        timestamp="20260305T000010",
        previous_production_run_identity=prev_identity,
        train_run_dir=Path("experiments") / "training" / "train-1",
    )

    assert validate_calls == ["production"]
    assert result["run_identity"]["promotion_id"] == "prom-1"
    assert result["decision"]["beats_previous"] is True
    assert result["promotion_thresholds_hash"] == "threshold-hash-1"
    assert result["context"]["promotion_conda_env_hash"] == "env-hash-prom"


def test_prepare_metadata_staging_validates_with_staging_and_omits_beats_previous(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build staging metadata and validate using staging schema branch."""
    args = _args(stage="staging")
    thresholds = cast(Any, SimpleNamespace(model_dump=lambda: {"thresholds": {"val": {"f1": 0.7}}}))
    prev_identity = PreviousProductionRunIdentity(None, None, None, None, None)

    validate_calls: list[str] = []

    monkeypatch.setattr("ml.promotion.persistence.prepare.get_conda_env_export", lambda: "env export")
    monkeypatch.setattr("ml.promotion.persistence.prepare.hash_environment", lambda _: "env-hash-prom")
    monkeypatch.setattr("ml.promotion.persistence.prepare.get_training_conda_env_hash", lambda _: "env-hash-prom")
    monkeypatch.setattr("ml.promotion.persistence.prepare.hash_thresholds", lambda _: "threshold-hash-1")

    def _validate(metadata: dict, stage: str) -> Any:
        validate_calls.append(stage)
        return SimpleNamespace(model_dump=lambda **kwargs: metadata)

    monkeypatch.setattr("ml.promotion.persistence.prepare.validate_promotion_metadata", _validate)

    result = prepare_metadata(
        run_id="stage-1",
        args=args,
        metrics={"val": {"f1": 0.82}},
        previous_production_metrics=None,
        promotion_thresholds=thresholds,
        promoted=True,
        beats_previous=False,
        reason="staged",
        git_commit="commit-abc",
        timestamp="20260305T000011",
        previous_production_run_identity=prev_identity,
        train_run_dir=Path("experiments") / "training" / "train-1",
    )

    assert validate_calls == ["staging"]
    assert result["run_identity"]["staging_id"] == "stage-1"
    assert "beats_previous" not in result["decision"]


def test_prepare_metadata_logs_warning_when_env_hashes_do_not_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emit reproducibility warning when promotion and training environment hashes differ."""
    args = _args(stage="staging")
    thresholds = cast(Any, SimpleNamespace(model_dump=lambda: {"thresholds": {"val": {"f1": 0.7}}}))
    prev_identity = PreviousProductionRunIdentity(None, None, None, None, None)
    warnings: list[str] = []

    monkeypatch.setattr("ml.promotion.persistence.prepare.get_conda_env_export", lambda: "env export")
    monkeypatch.setattr("ml.promotion.persistence.prepare.hash_environment", lambda _: "env-hash-prom")
    monkeypatch.setattr("ml.promotion.persistence.prepare.get_training_conda_env_hash", lambda _: "env-hash-train")
    monkeypatch.setattr("ml.promotion.persistence.prepare.hash_thresholds", lambda _: "threshold-hash-1")
    monkeypatch.setattr("ml.promotion.persistence.prepare.validate_promotion_metadata", lambda metadata, stage: SimpleNamespace(model_dump=lambda **kwargs: metadata))
    monkeypatch.setattr("ml.promotion.persistence.prepare.logger.warning", lambda msg: warnings.append(msg))

    prepare_metadata(
        run_id="stage-1",
        args=args,
        metrics={"val": {"f1": 0.82}},
        previous_production_metrics=None,
        promotion_thresholds=thresholds,
        promoted=False,
        beats_previous=False,
        reason="not promoted",
        git_commit="commit-abc",
        timestamp="20260305T000012",
        previous_production_run_identity=prev_identity,
        train_run_dir=Path("experiments") / "training" / "train-1",
    )

    assert warnings
    assert "does not match conda environment hash" in warnings[0]


def test_prepare_metadata_raises_for_invalid_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise UserError when stage is outside supported staging/production values."""
    args = _args(stage="invalid")
    thresholds = cast(Any, SimpleNamespace(model_dump=lambda: {"thresholds": {"val": {"f1": 0.7}}}))
    prev_identity = PreviousProductionRunIdentity(None, None, None, None, None)

    monkeypatch.setattr("ml.promotion.persistence.prepare.get_conda_env_export", lambda: "env export")
    monkeypatch.setattr("ml.promotion.persistence.prepare.hash_environment", lambda _: "env-hash-prom")
    monkeypatch.setattr("ml.promotion.persistence.prepare.get_training_conda_env_hash", lambda _: "env-hash-prom")
    monkeypatch.setattr("ml.promotion.persistence.prepare.hash_thresholds", lambda _: "threshold-hash-1")

    with pytest.raises(UserError, match="Stage must be either 'staging' or 'production'"):
        prepare_metadata(
            run_id="run-1",
            args=args,
            metrics={"val": {"f1": 0.82}},
            previous_production_metrics=None,
            promotion_thresholds=thresholds,
            promoted=False,
            beats_previous=False,
            reason="n/a",
            git_commit="commit-abc",
            timestamp="20260305T000013",
            previous_production_run_identity=prev_identity,
            train_run_dir=Path("experiments") / "training" / "train-1",
        )
