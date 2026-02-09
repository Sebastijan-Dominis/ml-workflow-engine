import logging
from pathlib import Path

from ml.utils.features.loading.latest_snapshot import get_latest_snapshot

logger = logging.getLogger(__name__)

def get_best_params_path(experiment_id: str, experiments_dir: Path) -> Path:
    if experiment_id == "latest":
        latest_experiment = get_latest_snapshot(experiments_dir)
        logger.info(f"Auto-resolved latest experiment ID: {latest_experiment}")
        return latest_experiment
    return experiments_dir / experiment_id