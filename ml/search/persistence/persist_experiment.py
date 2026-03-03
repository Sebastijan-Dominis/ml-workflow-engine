"""Persistence orchestration for search experiment metadata and runtime."""

import logging
from pathlib import Path

from ml.config.schemas.model_cfg import SearchModelConfig
from ml.types.splits import AllSplitsInfo
from ml.search.persistence.prepare_metadata import prepare_metadata
from ml.utils.persistence.save_metadata import save_metadata
from ml.utils.runtime.save_runtime import save_runtime_snapshot

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

def persist_experiment(
    model_cfg: SearchModelConfig, 
    *, 
    search_results: dict, 
    owner: str, 
    experiment_id: str, 
    search_dir: Path, 
    timestamp: str, 
    start_time: float, 
    feature_lineage: list[dict], 
    pipeline_hash: str,
    scoring_method: str,
    splits_info: AllSplitsInfo,
    overwrite_existing: bool = False
) -> None:
    """Persist search metadata and runtime snapshot for an experiment run.

    Args:
        model_cfg: Validated search model configuration.
        search_results: Search results payload.
        owner: Owner identifier for the experiment.
        experiment_id: Experiment identifier.
        search_dir: Target experiment directory.
        timestamp: Run timestamp string.
        start_time: Process start time used for runtime metadata.
        feature_lineage: Feature lineage records.
        pipeline_hash: Pipeline hash fingerprint.
        scoring_method: Scoring method used during search.
        splits_info: Dataset split information.
        overwrite_existing: Whether existing metadata/runtime should be overwritten.

    Returns:
        None.
    """
    
    metadata = prepare_metadata(
        model_cfg, 
        search_results=search_results, 
        owner=owner, 
        experiment_id=experiment_id, 
        timestamp=timestamp, 
        feature_lineage=feature_lineage, 
        pipeline_hash=pipeline_hash,
        scoring_method=scoring_method,
        splits_info=splits_info
    )

    logger.debug(f"Persisting experiment with overwrite_existing={overwrite_existing}.")

    save_metadata(
        metadata=metadata, 
        target_dir=search_dir,
        overwrite_existing=overwrite_existing
    )

    save_runtime_snapshot(
        target_dir=search_dir, 
        timestamp=timestamp, 
        hardware_info=model_cfg.search.hardware, 
        start_time=start_time,
        overwrite_existing=overwrite_existing
    )