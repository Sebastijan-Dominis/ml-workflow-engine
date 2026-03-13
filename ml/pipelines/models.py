"""Pipeline config Pydantic models for validation."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from ml.exceptions import ConfigError

VALID_STEPS = {"SchemaValidator", "FillCategoricalMissing", "FeatureEngineer", "FeatureSelector", "Model"}


class LineageConfig(BaseModel):
    created_by: str = Field(..., description="Author of the pipeline config")
    created_at: datetime = Field(..., description="Timestamp of config creation")


class PipelineConfig(BaseModel):
    """Pydantic schema for pipeline configurations."""

    name: str = Field(..., description="Name of the pipeline")
    version: str = Field(..., description="Version of the pipeline (e.g., v1)")
    description: str | None = Field(None, description="Optional description of the pipeline")
    steps: list[str] = Field(..., description="Ordered list of pipeline steps")
    assumptions: dict = Field(..., description="Assumptions about supported tasks and categorical handling")
    lineage: LineageConfig

    @field_validator("version")
    def check_version_format(cls, v: str) -> str:
        """Ensure version starts with 'v' followed by a number."""
        if not v.startswith("v") or not v[1:].isdigit():
            raise ConfigError(
                f"Pipeline version '{v}' must start with 'v' followed by a number (e.g., v1)"
            )
        return v

    @field_validator("steps")
    def check_steps_valid(cls, v: list[str]) -> list[str]:
        """Ensure all pipeline steps are known and non-empty."""
        if not v:
            raise ConfigError("Pipeline steps cannot be empty")
        unknown_steps = set(v) - VALID_STEPS
        if unknown_steps:
            raise ConfigError(f"Unknown pipeline steps: {unknown_steps}")
        return v

    @model_validator(mode="before")
    def validate_assumptions_keys(cls, values: dict) -> dict:
        """Ensure assumptions contain all required keys."""
        assumptions = values.get("assumptions", {})
        required_keys = {"handles_categoricals", "supports_regression", "supports_classification"}
        missing_keys = required_keys - assumptions.keys()
        if missing_keys:
            raise ConfigError(f"Pipeline assumptions missing keys: {missing_keys}")
        return values
