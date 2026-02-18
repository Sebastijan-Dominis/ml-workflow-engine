import logging
from pathlib import Path

from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.search.persistence.prepare_metadata import prepare_metadata
from ml.utils.persistence.save_metadata import save_metadata
from ml.utils.runtime.save_runtime import save_runtime_snapshot

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

def save_experiment(model_cfg: SearchModelConfig, *, search_results: dict, owner: str, experiment_id: str, search_dir: Path, timestamp: str, start_time: float, feature_lineage: list[dict], pipeline_hash: str) -> None:
    
    metadata = prepare_metadata(
        model_cfg, 
        search_results=search_results, 
        owner=owner, 
        experiment_id=experiment_id, 
        search_dir=search_dir, 
        timestamp=timestamp, 
        feature_lineage=feature_lineage, 
        pipeline_hash=pipeline_hash
    )

    save_metadata(metadata=metadata, target_dir=search_dir)

    save_runtime_snapshot(
        target_dir=search_dir, 
        timestamp=timestamp, 
        hardware_info=model_cfg.search.hardware, 
        start_time=start_time
    )