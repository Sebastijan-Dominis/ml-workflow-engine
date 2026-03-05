"""Defines data models for representing search results and related metadata in a structured format."""
from pydantic import BaseModel, ConfigDict


class Phases(BaseModel):
    """Represents the different phases of the search process."""
    broad: dict
    narrow: dict | None = None

    model_config = ConfigDict(extra="forbid")  # Pydantic options for strict schema validation behavior

class SearchResults(BaseModel):
    """Structured representation of search results, including best parameters and phase details."""
    best_pipeline_params: dict
    best_model_params: dict
    phases: Phases
