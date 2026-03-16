"""A module containing an example pipeline configuration in YAML format, which can be used as a template for creating new pipeline configurations in the ML service."""

EXAMPLE_CONFIG = """name: tabular_catboost_v2
version: v2
description: >
  Example pipeline config for tabular data using CatBoost.
  This config includes steps for handling categorical features,
  feature engineering, and model training. It serves as a
  template for creating new pipeline configurations.
steps:
  - SchemaValidator
  - FillCategoricalMissing
  - FeatureEngineer
  - FeatureSelector
  - Model

assumptions:
  handles_categoricals: true
  supports_regression: true
  supports_classification: true

lineage:
  created_by: Name Surname
"""
