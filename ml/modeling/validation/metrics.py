"""This module contains functions for validating training and evaluation metrics against their respective Pydantic models. It ensures that the metrics conform to the expected structure and types defined in the TrainingMetrics and EvaluationMetrics models. If the validation fails, it raises a RuntimeMLError with an appropriate error message."""
import logging

from ml.exceptions import RuntimeMLError
from ml.modeling.models.metrics import EvaluationMetrics, TrainingMetrics

logger = logging.getLogger(__name__)

def validate_training_metrics(training_metrics_raw: dict) -> TrainingMetrics:
    """Validate the training metrics against the TrainingMetrics model.

    Args:
        training_metrics_raw: A dictionary containing the training metrics.

    Returns:
        TrainingMetrics: A validated TrainingMetrics object.

    Raises:
        RuntimeMLError: If the training metrics are invalid.
    """
    try:
        training_metrics = TrainingMetrics(**training_metrics_raw)
        logger.debug("Validated training metrics.")
        return training_metrics
    except Exception as e:
        msg = "Error validating training metrics."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e

def validate_evaluation_metrics(evaluation_metrics_raw: dict) -> EvaluationMetrics:
    """Validate the evaluation metrics against the EvaluationMetrics model.

    Args:
        evaluation_metrics_raw: A dictionary containing the evaluation metrics.

    Returns:
        EvaluationMetrics: A validated EvaluationMetrics object.

    Raises:
        RuntimeMLError: If the evaluation metrics are invalid.
    """
    try:
        evaluation_metrics = EvaluationMetrics(**evaluation_metrics_raw)
        logger.debug("Validated evaluation metrics.")
        return evaluation_metrics
    except Exception as e:
        msg = "Error validating evaluation metrics."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e
