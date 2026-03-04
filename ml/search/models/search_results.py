"""Defines data models for representing search results and related metadata in a structured format."""
from pydantic import BaseModel


class Phases(BaseModel):
    """Represents the different phases of the search process."""
    broad: dict
    narrow: dict | None = None

    class Config:
        extra = "forbid"

class SearchResults(BaseModel):
    """Structured representation of search results, including best parameters and phase details."""
    best_pipeline_params: dict
    best_model_params: dict
    phases: Phases
