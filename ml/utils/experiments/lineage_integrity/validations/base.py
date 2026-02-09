import logging
from pathlib import Path

from ml.exceptions import PipelineContractError

logger = logging.getLogger(__name__)

def validate_base_lineage_integrity(experiment_dir: Path) -> None:
    experiment_json_path = experiment_dir / "experiment.json"
    if not experiment_json_path.exists():
        msg = f"Lineage integrity validation failed: {experiment_json_path} does not exist."
        logger.error(msg)
        raise PipelineContractError(msg)
    runtime_json_path = experiment_dir / "runtime.json"
    if not runtime_json_path.exists():
        msg = f"Lineage integrity validation failed: {runtime_json_path} does not exist."
        logger.error(msg)
        raise PipelineContractError(msg)
    
    logger.debug(f"Base lineage integrity validation passed: Found {experiment_json_path} and {runtime_json_path}.")
