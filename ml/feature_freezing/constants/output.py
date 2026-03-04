"""Output models shared by feature-freezing strategies."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FreezeOutput:
    """Result payload returned by a feature-freezing strategy run."""

    snapshot_path: Path
    metadata: dict
