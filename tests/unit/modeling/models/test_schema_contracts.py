"""Unit tests for modeling schema contracts and invariants."""

from __future__ import annotations

from typing import Any

import pytest
from ml.modeling.models.artifacts import Artifacts
from ml.modeling.models.config_fingerprint import ConfigFingerprint
from ml.modeling.models.experiment_lineage import ExperimentLineage
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.modeling.models.run_identity import RunIdentity
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _feature_lineage_payload() -> dict[str, str]:
    """Return a valid feature-lineage payload reusable across model contract tests."""
    return {
        "name": "booking_context_features",
        "version": "v1",
        "snapshot_id": "snapshot-2026-03-06",
        "file_hash": "file-hash-123",
        "in_memory_hash": "memory-hash-123",
        "feature_schema_hash": "schema-hash-123",
        "operator_hash": "operator-hash-123",
        "feature_type": "tabular",
        "file_name": "features.py",
        "data_format": "parquet",
    }


def test_run_identity_accepts_success_status() -> None:
    """Accept the only allowed status literal for run-identity objects."""
    run_identity = RunIdentity(
        train_run_id="train-001",
        snapshot_id="snapshot-001",
        status="success",
    )

    assert run_identity.status == "success"


def test_run_identity_rejects_non_success_status_values() -> None:
    """Reject invalid status values to keep run-state semantics strict."""
    with pytest.raises(ValidationError, match="status"):
        RunIdentity.model_validate(
            {
                "train_run_id": "train-001",
                "snapshot_id": "snapshot-001",
                "status": "failed",
            }
        )


def test_config_fingerprint_defaults_pipeline_hash_to_empty_string() -> None:
    """Use a stable empty-string default for optional pipeline fingerprint values."""
    fingerprint = ConfigFingerprint(config_hash="cfg-hash-001")

    assert fingerprint.config_hash == "cfg-hash-001"
    assert fingerprint.pipeline_cfg_hash == ""


def test_artifacts_accepts_model_only_payload_and_defaults_optional_pipeline_fields() -> None:
    """Permit artifacts records that only include model fields when pipeline is absent."""
    artifacts = Artifacts(model_hash="model-hash-001", model_path="artifacts/model.cbm")

    assert artifacts.model_hash == "model-hash-001"
    assert artifacts.pipeline_path is None
    assert artifacts.pipeline_hash is None


def test_feature_lineage_rejects_unsupported_feature_type_literal() -> None:
    """Enforce feature-type literal domain to protect downstream assumptions."""
    payload = _feature_lineage_payload()
    payload["feature_type"] = "graph"

    with pytest.raises(ValidationError, match="feature_type"):
        FeatureLineage.model_validate(payload)


def test_experiment_lineage_constructs_nested_feature_lineage_models_from_dicts() -> None:
    """Coerce nested lineage dict payloads into typed ``FeatureLineage`` model instances."""
    payload: dict[str, Any] = {
        "feature_lineage": [_feature_lineage_payload()],
        "target_column": "is_canceled",
        "problem": "cancellation",
        "segment": "city_hotel",
        "model_version": "v1",
    }

    lineage = ExperimentLineage.model_validate(payload)

    assert len(lineage.feature_lineage) == 1
    assert isinstance(lineage.feature_lineage[0], FeatureLineage)
    assert lineage.feature_lineage[0].name == "booking_context_features"


def test_experiment_lineage_rejects_invalid_nested_feature_lineage_payloads() -> None:
    """Fail validation when nested feature-lineage entries are structurally invalid."""
    invalid_child = _feature_lineage_payload()
    invalid_child.pop("name")

    with pytest.raises(ValidationError, match="feature_lineage"):
        ExperimentLineage.model_validate(
            {
                "feature_lineage": [invalid_child],
                "target_column": "is_canceled",
                "problem": "cancellation",
                "segment": "city_hotel",
                "model_version": "v1",
            }
        )
