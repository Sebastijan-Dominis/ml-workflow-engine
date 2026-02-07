import json
from pathlib import Path

from ml.utils.runtime.runtime_snapshot import build_runtime_snapshot

def save_runtime_snapshot(run_dir: Path, timestamp: str) -> None:
    snapshot = build_runtime_snapshot(timestamp)
    snapshot_path = run_dir / "runtime.json"
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=4)