"""Typed container models for schema-derived pipeline feature groups."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineFeatures:
    """Feature groups used during pipeline assembly and validation."""

    input_features: list[str]
    derived_features: list[str]
    categorical_features: list[str]
    selected_features: list[str]