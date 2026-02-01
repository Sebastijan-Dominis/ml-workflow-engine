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


class FeatureSetConfig(BaseModel):
    ref: str
    name: str
    version: str
    schema_format: str
    input_schema: str
    derived_schema: str
    data_format: str
    X_train: str
    X_val: str
    X_test: str
    y_train: str
    y_val: str
    y_test: str


class FeatureStoreConfig(BaseModel):
    path: str
    feature_sets: list[FeatureSetConfig]


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
    feature_store: FeatureStoreConfig
    data_type: Optional[str] = None
