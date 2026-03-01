class MLException(Exception):
    """Base class for all ML pipeline errors."""

class UserError(MLException):
    """Errors caused by user configuration or misuse."""

class RuntimeMLException(MLException):
    """Errors caused by internal failures."""

class ConfigError(UserError):
    """Invalid or inconsistent configuration."""

class DataError(UserError):
    """Feature store or data issues."""

class PipelineContractError(UserError):
    """Violations of structural or logical expectations between pipeline
    stages or experiment components, including incompatible artifacts,
    lineage inconsistencies, incorrect stage ordering, or execution under
    an unrelated or incompatible experiment context."""

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
