from ml.exceptions import (
    UserError,
    RuntimeMLException,
    ConfigError,
    DataError,
    PipelineContractError,
    SearchError,
    TrainingError,
    EvaluationError,
    ExplainabilityError,
    PersistenceError,
)

from ml.cli.exit_codes import (
    EXIT_UNEXPECTED_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_DATA_ERROR,
    EXIT_PIPELINE_ERROR,
    EXIT_SEARCH_ERROR,
    EXIT_TRAINING_ERROR,
    EXIT_EVALUATION_ERROR,
    EXIT_EXPLAINABILITY_ERROR,
    EXIT_PERSISTENCE_ERROR,
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
}

def resolve_exit_code(exc: Exception) -> int:
    """
    Convert an exception into a CLI exit code.
    """
    for exc_type, code in EXCEPTION_EXIT_CODE_MAP.items():
        if isinstance(exc, exc_type):
            return code
    if isinstance(exc, UserError):
        return EXIT_CONFIG_ERROR
    if isinstance(exc, RuntimeMLException):
        return EXIT_UNEXPECTED_ERROR
    return EXIT_UNEXPECTED_ERROR
