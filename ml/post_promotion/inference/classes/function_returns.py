"""A module for defining function return types in the inference pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass
class ArtifactLoadingReturn:
    """Return type for the artifact loading function."""
    artifact: Any
    artifact_hash: str
    artifact_type: Literal["pipeline", "model"]

@dataclass
class PredictionStoringReturn:
    """Return type for the prediction storing function."""
    file_path: Path
    cols: list[str]
