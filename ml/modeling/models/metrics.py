"""This module defines the data models for training and evaluation metrics using Pydantic."""
from pydantic import BaseModel


class Metrics(BaseModel):
    """Base model for metrics data."""
    task_type: str
    algorithm: str

class TrainingMetrics(Metrics):
    """Model training metrics."""
    metrics: dict[str, float] | dict[str, dict[str, float]]

class EvaluationMetrics(Metrics):
    """Model evaluation metrics."""
    train: dict
    val: dict
    test: dict
