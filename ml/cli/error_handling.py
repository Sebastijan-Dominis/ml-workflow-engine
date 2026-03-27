"""CLI exception-to-exit-code mapping utilities.

This module centralizes translation of domain-specific exceptions into stable
process exit codes so command-line entrypoints can fail consistently.
"""

from ml.cli.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_DATA_ERROR,
    EXIT_EVALUATION_ERROR,
    EXIT_EXPLAINABILITY_ERROR,
    EXIT_INFERENCE_ERROR,
    EXIT_MONITORING_ERROR,
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
    InferenceError,
    MonitoringError,
    PersistenceError,
    PipelineContractError,
    RuntimeMLError,
    SearchError,
    TrainingError,
    UserError,
)

EXCEPTION_EXIT_CODE_MAP = {
    ConfigError: EXIT_CONFIG_ERROR,
    DataError: EXIT_DATA_ERROR,
    PipelineContractError: EXIT_PIPELINE_ERROR,
    SearchError: EXIT_SEARCH_ERROR,
    TrainingError: EXIT_TRAINING_ERROR,
    EvaluationError: EXIT_EVALUATION_ERROR,
    ExplainabilityError: EXIT_EXPLAINABILITY_ERROR,
    PersistenceError: EXIT_PERSISTENCE_ERROR,
    InferenceError: EXIT_INFERENCE_ERROR,
    MonitoringError: EXIT_MONITORING_ERROR,

}

def resolve_exit_code(exc: Exception) -> int:
    """Resolve a process exit code from an exception instance.

    Resolution order:
        1. Exact/derived matches from ``EXCEPTION_EXIT_CODE_MAP``.
        2. ``UserError`` fallback to configuration error exit code.
        3. ``RuntimeMLError`` fallback to unexpected error exit code.
        4. Default unexpected error exit code for all other exceptions.

    Args:
        exc: Exception raised during CLI execution.

    Returns:
        int: Exit code representing the error category.
    """
    for exc_type, code in EXCEPTION_EXIT_CODE_MAP.items():
        if isinstance(exc, exc_type):
            return code
    if isinstance(exc, UserError):
        return EXIT_CONFIG_ERROR
    if isinstance(exc, RuntimeMLError):
        return EXIT_UNEXPECTED_ERROR
    return EXIT_UNEXPECTED_ERROR
