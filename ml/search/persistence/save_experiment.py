import logging
from pathlib import Path

from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.search.persistence.save_metadata import save_metadata
from ml.utils.runtime.save_runtime import save_runtime_snapshot

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

def save_experiment(model_cfg: SearchModelConfig, search_results: dict, owner: str, *, experiment_id: str, timestamp: str, start_time: float, feature_lineage: list[dict], pipeline_hash: str) -> None:
    run_dir = EXPERIMENTS_DIR / model_cfg.problem / model_cfg.segment.name / model_cfg.version / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_metadata(model_cfg, search_results, owner, experiment_id=experiment_id, timestamp=timestamp, feature_lineage=feature_lineage, pipeline_hash=pipeline_hash)
    save_runtime_snapshot(run_dir, timestamp, model_cfg.search.hardware, start_time=start_time)