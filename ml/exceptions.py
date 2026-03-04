"""Project-wide exception hierarchy for ML workflows.

The hierarchy separates user-actionable issues from internal runtime failures,
enabling consistent error handling, logging, and CLI exit-code mapping.
"""

class MLBaseError(Exception):
    """Base class for all ML pipeline errors."""

class UserError(MLBaseError):
    """Errors caused by user configuration or misuse."""

class RuntimeMLError(MLBaseError):
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

class SearchError(RuntimeMLError):
    """Hyperparameter search failure."""

class TrainingError(RuntimeMLError):
    """Model training failure."""

class EvaluationError(RuntimeMLError):
    """Evaluation or metric computation failure."""

class ExplainabilityError(RuntimeMLError):
    """Explainability or interpretation stage failure."""

class PersistenceError(RuntimeMLError):
    """Experiment or artifact saving failure."""
