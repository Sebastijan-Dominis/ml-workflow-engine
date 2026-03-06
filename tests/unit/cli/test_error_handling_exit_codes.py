"""Unit tests for CLI exception-to-exit-code mapping utilities."""

from __future__ import annotations

import pytest
from ml.cli.error_handling import resolve_exit_code
from ml.cli.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_DATA_ERROR,
    EXIT_EVALUATION_ERROR,
    EXIT_EXPLAINABILITY_ERROR,
    EXIT_PERSISTENCE_ERROR,
    EXIT_PIPELINE_ERROR,
    EXIT_SEARCH_ERROR,
    EXIT_TRAINING_ERROR,
    EXIT_UNEXPECTED_ERROR,
)
from ml.exceptions import (
    ConfigError,
    DataError,
    EvaluationError,
    ExplainabilityError,
    PersistenceError,
    PipelineContractError,
    RuntimeMLError,
    SearchError,
    TrainingError,
    UserError,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ConfigError("bad config"), EXIT_CONFIG_ERROR),
        (DataError("bad data"), EXIT_DATA_ERROR),
        (PipelineContractError("contract broken"), EXIT_PIPELINE_ERROR),
        (SearchError("search failed"), EXIT_SEARCH_ERROR),
        (TrainingError("training failed"), EXIT_TRAINING_ERROR),
        (EvaluationError("evaluation failed"), EXIT_EVALUATION_ERROR),
        (ExplainabilityError("explain failed"), EXIT_EXPLAINABILITY_ERROR),
        (PersistenceError("persist failed"), EXIT_PERSISTENCE_ERROR),
    ],
)
def test_resolve_exit_code_maps_domain_exceptions_to_expected_codes(
    exc: Exception,
    expected: int,
) -> None:
    """Map each domain exception class to its canonical CLI exit code."""
    assert resolve_exit_code(exc) == expected


def test_resolve_exit_code_maps_user_error_to_config_code() -> None:
    """Treat generic UserError as a configuration-class failure."""
    assert resolve_exit_code(UserError("user mistake")) == EXIT_CONFIG_ERROR


def test_resolve_exit_code_maps_runtime_error_to_unexpected_code() -> None:
    """Map uncategorized runtime pipeline failures to unexpected error code."""
    assert resolve_exit_code(RuntimeMLError("runtime failure")) == EXIT_UNEXPECTED_ERROR


def test_resolve_exit_code_defaults_unknown_exceptions_to_unexpected_code() -> None:
    """Fallback unknown Python exceptions to generic unexpected error code."""
    assert resolve_exit_code(ValueError("boom")) == EXIT_UNEXPECTED_ERROR


def test_resolve_exit_code_prefers_specific_mapping_over_user_error_fallback() -> None:
    """Use specific mapped codes for UserError subclasses before generic fallback."""
    assert resolve_exit_code(PipelineContractError("pipeline mismatch")) == EXIT_PIPELINE_ERROR
