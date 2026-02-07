import json
from pathlib import Path

def get_metadata(latest_snapshot: Path) -> dict:
    metadata_path = latest_snapshot / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata