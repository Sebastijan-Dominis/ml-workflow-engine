from dataclasses import dataclass
from pathlib import Path


@dataclass
class FreezeOutput:
    snapshot_path: Path
    metadata: dict