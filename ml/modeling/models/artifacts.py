"""Artifacts model for tracking model and pipeline artifacts."""

from pydantic import BaseModel


class Artifacts(BaseModel):
    """Model representing the artifacts produced during training or evaluation, including paths and hashes for integrity verification."""
    model_hash: str
    model_path: str
    pipeline_path: str | None = None
    pipeline_hash: str | None = None
