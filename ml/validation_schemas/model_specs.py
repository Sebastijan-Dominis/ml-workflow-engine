"""Pydantic schemas for binary classification CatBoost training configs.

This module defines typed configuration models used to validate YAML
training configuration files. Using Pydantic ensures the training code
receives a well-structured dictionary with expected keys and types.
"""

from pydantic import BaseModel
from typing import Optional


class SegmentConfig(BaseModel):
    name: str
    description: Optional[str] = None


class FeaturesConfig(BaseModel):
    """Data-related configuration values.

    Attributes:
        engineered_features: List of engineered feature column names.
        engineered_categorical_features: List of engineered categorical feature column names.
        raw_schema: Path to data schema file.
        features_path: Path to directory containing feature parquet files.
        features_version: Semantic version or identifier for feature set.
        target: Name of the target column.
        X_train, X_val, X_test: Filenames for feature tables.
        y_train, y_val, y_test: Filenames for label tables.
    """

    version: str
    path: str
    raw_schema: str
    derived_schema: str
    X_train: str
    X_val: str
    X_test: str
    y_train: str
    y_val: str
    y_test: str


class FeatureEngineeringConfig(BaseModel):
    enabled: bool
    operators: Optional[list[str]] = None


class ModelSpecsSchema(BaseModel):
    """Root schema that represents the full training configuration."""

    problem: str
    segment: SegmentConfig
    version: str
    task: str
    target: str
    algorithm: str
    model_class: str
    pipeline: str
    seed: int
    features: FeaturesConfig
    feature_engineering: FeatureEngineeringConfig
    search_version: str
