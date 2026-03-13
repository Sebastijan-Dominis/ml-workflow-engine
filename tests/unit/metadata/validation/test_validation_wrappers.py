"""Unit tests for metadata validation wrapper functions.

These tests verify that wrapper functions:
1. call the intended schema validator,
2. return validated objects unchanged on success, and
3. wrap downstream validation failures into project-specific exception types.
"""

import importlib
import sys
import types

import pytest
from ml.exceptions import RuntimeMLError

# Some metadata schemas import ml.types, which imports catboost at module import time.
if "catboost" not in sys.modules:
    catboost_stub = types.ModuleType("catboost")
    catboost_stub.__dict__.update(
        {
            "CatBoostClassifier": type("CatBoostClassifier", (), {}),
            "CatBoostRegressor": type("CatBoostRegressor", (), {}),
        }
    )
    sys.modules["catboost"] = catboost_stub

validate_interim_dataset_metadata = importlib.import_module(
    "ml.metadata.validation.data.interim"
).validate_interim_dataset_metadata
validate_processed_dataset_metadata = importlib.import_module(
    "ml.metadata.validation.data.processed"
).validate_processed_dataset_metadata
validate_raw_snapshot_metadata = importlib.import_module(
    "ml.metadata.validation.data.raw"
).validate_raw_snapshot_metadata
validate_freeze_metadata = importlib.import_module(
    "ml.metadata.validation.features.feature_freezing"
).validate_freeze_metadata
validate_promotion_metadata = importlib.import_module(
    "ml.metadata.validation.promotion.promote"
).validate_promotion_metadata
validate_evaluation_metadata = importlib.import_module(
    "ml.metadata.validation.runners.evaluation"
).validate_evaluation_metadata
validate_explainability_metadata = importlib.import_module(
    "ml.metadata.validation.runners.explainability"
).validate_explainability_metadata
validate_training_metadata = importlib.import_module(
    "ml.metadata.validation.runners.training"
).validate_training_metadata
validate_search_record = importlib.import_module(
    "ml.metadata.validation.search.search"
).validate_search_record


pytestmark = pytest.mark.unit


def test_validate_raw_snapshot_metadata_returns_validated_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure raw snapshot metadata wrapper returns schema output on successful validation."""

    expected = {"ok": True}

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            assert payload == {"raw": 1}
            return expected

    monkeypatch.setattr("ml.metadata.validation.data.raw.RawSnapshotMetadata", _Schema)

    result = validate_raw_snapshot_metadata({"raw": 1})

    assert result is expected


def test_validate_raw_snapshot_metadata_wraps_schema_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure raw snapshot metadata wrapper converts schema failures to RuntimeMLError."""

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            raise ValueError("invalid")

    monkeypatch.setattr("ml.metadata.validation.data.raw.RawSnapshotMetadata", _Schema)

    with pytest.raises(RuntimeMLError, match="Error validating raw snapshot metadata"):
        validate_raw_snapshot_metadata({"raw": 1})


def test_validate_interim_dataset_metadata_wraps_schema_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure interim metadata wrapper converts schema failures to RuntimeMLError."""

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            raise ValueError("invalid")

    monkeypatch.setattr("ml.metadata.validation.data.interim.InterimDatasetMetadata", _Schema)

    with pytest.raises(RuntimeMLError, match="Error validating interim dataset metadata"):
        validate_interim_dataset_metadata({"x": 1})


def test_validate_interim_dataset_metadata_returns_validated_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure interim metadata wrapper returns schema output on success."""

    expected = {"kind": "interim"}

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            assert payload == {"x": 1}
            return expected

    monkeypatch.setattr("ml.metadata.validation.data.interim.InterimDatasetMetadata", _Schema)

    result = validate_interim_dataset_metadata({"x": 1})

    assert result is expected


def test_validate_processed_dataset_metadata_wraps_schema_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure processed metadata wrapper converts schema failures to RuntimeMLError."""

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            raise ValueError("invalid")

    monkeypatch.setattr("ml.metadata.validation.data.processed.ProcessedDatasetMetadata", _Schema)

    with pytest.raises(RuntimeMLError, match="Error validating processed dataset metadata"):
        validate_processed_dataset_metadata({"x": 1})


def test_validate_processed_dataset_metadata_returns_validated_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure processed metadata wrapper returns schema output on success."""

    expected = {"kind": "processed"}

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            assert payload == {"x": 1}
            return expected

    monkeypatch.setattr("ml.metadata.validation.data.processed.ProcessedDatasetMetadata", _Schema)

    result = validate_processed_dataset_metadata({"x": 1})

    assert result is expected


def test_validate_freeze_metadata_wraps_schema_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure freeze metadata wrapper converts schema failures to RuntimeMLError."""

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            raise ValueError("invalid")

    monkeypatch.setattr("ml.metadata.validation.features.feature_freezing.FreezeMetadata", _Schema)

    with pytest.raises(RuntimeMLError, match="Freeze metadata validation failed"):
        validate_freeze_metadata({"x": 1})


def test_validate_freeze_metadata_returns_validated_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure freeze metadata wrapper returns schema output on success."""

    expected = {"kind": "freeze"}

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            assert payload == {"x": 1}
            return expected

    monkeypatch.setattr("ml.metadata.validation.features.feature_freezing.FreezeMetadata", _Schema)

    result = validate_freeze_metadata({"x": 1})

    assert result is expected


def test_validate_search_record_wraps_schema_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure search record wrapper converts schema failures to RuntimeMLError."""

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            raise ValueError("invalid")

    monkeypatch.setattr("ml.metadata.validation.search.search.SearchRecord", _Schema)

    with pytest.raises(RuntimeMLError, match="Error validating search record"):
        validate_search_record({"x": 1})


def test_validate_search_record_returns_validated_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure search record wrapper returns schema output on success."""

    expected = {"kind": "search"}

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            assert payload == {"x": 1}
            return expected

    monkeypatch.setattr("ml.metadata.validation.search.search.SearchRecord", _Schema)

    result = validate_search_record({"x": 1})

    assert result is expected


def test_validate_training_metadata_wraps_schema_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure training metadata wrapper converts schema failures to RuntimeMLError."""

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            raise ValueError("invalid")

    monkeypatch.setattr("ml.metadata.validation.runners.training.TrainingMetadata", _Schema)

    with pytest.raises(RuntimeMLError, match="Training metadata validation failed"):
        validate_training_metadata({"x": 1})


def test_validate_training_metadata_returns_validated_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure training metadata wrapper returns schema output on success."""

    expected = {"kind": "training"}

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            assert payload == {"x": 1}
            return expected

    monkeypatch.setattr("ml.metadata.validation.runners.training.TrainingMetadata", _Schema)

    result = validate_training_metadata({"x": 1})

    assert result is expected


def test_validate_evaluation_metadata_wraps_schema_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure evaluation metadata wrapper converts schema failures to RuntimeMLError."""

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            raise ValueError("invalid")

    monkeypatch.setattr("ml.metadata.validation.runners.evaluation.EvaluationMetadata", _Schema)

    with pytest.raises(RuntimeMLError, match="Evaluation metadata validation failed"):
        validate_evaluation_metadata({"x": 1})


def test_validate_evaluation_metadata_returns_validated_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure evaluation metadata wrapper returns schema output on success."""

    expected = {"kind": "evaluation"}

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            assert payload == {"x": 1}
            return expected

    monkeypatch.setattr("ml.metadata.validation.runners.evaluation.EvaluationMetadata", _Schema)

    result = validate_evaluation_metadata({"x": 1})

    assert result is expected


def test_validate_explainability_metadata_wraps_schema_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure explainability metadata wrapper converts schema failures to RuntimeMLError."""

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            raise ValueError("invalid")

    monkeypatch.setattr("ml.metadata.validation.runners.explainability.ExplainabilityMetadata", _Schema)

    with pytest.raises(RuntimeMLError, match="Explainability metadata validation failed"):
        validate_explainability_metadata({"x": 1})


def test_validate_explainability_metadata_returns_validated_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure explainability metadata wrapper returns schema output on success."""

    expected = {"kind": "explainability"}

    class _Schema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            assert payload == {"x": 1}
            return expected

    monkeypatch.setattr("ml.metadata.validation.runners.explainability.ExplainabilityMetadata", _Schema)

    result = validate_explainability_metadata({"x": 1})

    assert result is expected


def test_validate_promotion_metadata_production_uses_production_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure promotion wrapper uses ProductionPromotionMetadata for production stage."""

    expected = {"stage": "production"}

    class _ProductionSchema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            return expected

    monkeypatch.setattr("ml.metadata.validation.promotion.promote.ProductionPromotionMetadata", _ProductionSchema)

    result = validate_promotion_metadata({"x": 1}, stage="production")

    assert result is expected


def test_validate_promotion_metadata_staging_uses_staging_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure promotion wrapper uses StagingPromotionMetadata for staging stage."""

    expected = {"stage": "staging"}

    class _StagingSchema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            return expected

    monkeypatch.setattr("ml.metadata.validation.promotion.promote.StagingPromotionMetadata", _StagingSchema)

    result = validate_promotion_metadata({"x": 1}, stage="staging")

    assert result is expected


def test_validate_promotion_metadata_wraps_production_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure promotion wrapper converts production schema failures to RuntimeMLError."""

    class _ProductionSchema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            raise ValueError("invalid")

    monkeypatch.setattr("ml.metadata.validation.promotion.promote.ProductionPromotionMetadata", _ProductionSchema)

    with pytest.raises(RuntimeMLError, match="Failed to validate promotion metadata against ProductionPromotionMetadata"):
        validate_promotion_metadata({"x": 1}, stage="production")


def test_validate_promotion_metadata_wraps_staging_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure promotion wrapper converts staging schema failures to RuntimeMLError."""

    class _StagingSchema:
        @staticmethod
        def model_validate(payload: dict) -> dict:
            raise ValueError("invalid")

    monkeypatch.setattr("ml.metadata.validation.promotion.promote.StagingPromotionMetadata", _StagingSchema)

    with pytest.raises(RuntimeMLError, match="Failed to validate promotion metadata against StagingPromotionMetadata"):
        validate_promotion_metadata({"x": 1}, stage="staging")
