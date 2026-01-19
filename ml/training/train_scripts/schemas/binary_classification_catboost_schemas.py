"""Pydantic schemas for binary classification CatBoost training configs.

This module defines typed configuration models used to validate YAML
training configuration files. Using Pydantic ensures the training code
receives a well-structured dictionary with expected keys and types.
"""

from pydantic import BaseModel
from typing import Optional


class DataConfig(BaseModel):
    """Data-related configuration values.

    Attributes:
        features_path: Path to directory containing feature parquet files.
        features_version: Semantic version or identifier for feature set.
        target: Name of the target column.
        train_file, val_file, test_file: Filenames for feature tables.
        y_train, y_val, y_test: Filenames for label tables.
    """

    features_path: str
    features_version: str
    target: str
    train_file: str
    val_file: str
    test_file: str
    y_train: str
    y_val: str
    y_test: str


class ModelParams(BaseModel):
    """Parameters passed to CatBoost model constructor.

    Fields are optional to allow sane defaults to be specified in training
    configs while still validating their presence and types when provided.
    """

    # Additional CatBoost parameters can be added as needed
    random_strength: Optional[int] = None
    min_data_in_leaf: Optional[int] = None
    learning_rate: Optional[float] = None
    l2_leaf_reg: Optional[float] = None
    depth: Optional[int] = None
    colsample_bylevel: Optional[float] = None
    border_count: Optional[int] = None
    bagging_temperature: Optional[float] = None

    # Default values for commonly used parameters
    iterations: Optional[int] = 2500
    task_type: Optional[str] = "CPU"
    random_state: Optional[int] = 42
    verbose: Optional[int] = 200


class ModelConfig(BaseModel):
    """Top-level model configuration.

    Attributes:
        algorithm: String identifier for the training algorithm.
        params: ``ModelParams`` instance.
        threshold: Prediction threshold used by downstream scoring.
    """

    algorithm: str
    params: ModelParams
    threshold: float


class PipelineConfig(BaseModel):
    """Flags controlling which preprocessing steps to include in pipeline."""

    validate_schema: bool
    fill_categorical_missing: bool
    feature_engineering: bool
    feature_selection: bool


class ExplainabilityConfig(BaseModel):
    """Explainability-related configuration options."""

    feature_importance_method: str
    shap_method: str


class ConfigSchema(BaseModel):
    """Root schema that represents the full training configuration."""

    name: str
    task: str
    version: str
    data: DataConfig
    model: ModelConfig
    pipeline: PipelineConfig
    explainability: ExplainabilityConfig
