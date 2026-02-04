class MLException(Exception):
    """Base class for all ML pipeline errors."""

class UserError(MLException):
    """Errors caused by user configuration or misuse."""

class RuntimeMLException(MLException):
    """Errors caused by internal failures."""

class ConfigError(UserError):
    """Invalid or inconsistent configuration."""


class DataError(UserError):
    """Feature store or dataset issues."""


class PipelineContractError(UserError):
    """Mismatch between model, features, or pipeline."""


class SearchError(RuntimeMLException):
    """Hyperparameter search failure."""


class TrainingError(RuntimeMLException):
    """Model training failure."""


class EvaluationError(RuntimeMLException):
    """Evaluation or metric computation failure."""


class ExplainabilityError(RuntimeMLException):
    """Explainability or interpretation stage failure."""


class PersistenceError(RuntimeMLException):
    """Experiment or artifact saving failure."""
