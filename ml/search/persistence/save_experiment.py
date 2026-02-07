import logging
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from typing import Optional

logger = logging.getLogger(__name__)

from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.search.persistence.save_metadata import save_metadata
from ml.search.persistence.save_runtime import save_runtime_snapshot

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

def save_experiment(model_cfg: SearchModelConfig, search_results: dict, owner: str, *, experiment_id: Optional[str] = None) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_id is None:
        experiment_id = f"{timestamp}_{uuid4().hex[:8]}"

    run_dir = EXPERIMENTS_DIR / model_cfg.problem / model_cfg.segment.name / model_cfg.version / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_metadata(model_cfg, search_results, owner, experiment_id=experiment_id, timestamp=timestamp)
    save_runtime_snapshot(run_dir, timestamp)