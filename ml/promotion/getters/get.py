import logging
from pathlib import Path

from ml.exceptions import PersistenceError, UserError
from ml.promotion.constants.constants import RunnersMetadata
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def get_runners_metadata(train_run_dir: Path, eval_run_dir: Path, explain_run_dir: Path) -> RunnersMetadata:
    train_metadata = load_json(train_run_dir / "metadata.json")
    eval_metadata = load_json(eval_run_dir / "metadata.json")
    explain_metadata = load_json(explain_run_dir / "metadata.json")
    return RunnersMetadata(train_metadata, eval_metadata, explain_metadata)



def extract_thresholds(promotion_thresholds: dict, problem: str, segment: str) -> dict:
    promotion_thresholds = promotion_thresholds.get(problem, {}).get(segment, {})
    if not promotion_thresholds:
        msg = f"No promotion thresholds found for problem={problem} segment={segment}"
        logger.error(msg)
        raise UserError(msg)
    
    return promotion_thresholds

def get_artifacts(explain_metadata: dict) -> dict:
    artifacts = explain_metadata.get("artifacts", {})

    if not artifacts or artifacts.get("model_hash") is None or artifacts.get("model_path") is None:
        msg = f"Explainability metadata is missing required artifact information. Artifacts found: {artifacts}"
        logger.error(msg)
        raise PersistenceError(msg)
    
    return artifacts

def get_feature_lineage(training_metadata: dict) -> list[str]:
    feature_lineage = training_metadata.get("lineage", {}).get("feature_lineage")
    if not feature_lineage:
        msg = "Training metadata is missing feature lineage information."
        logger.error(msg)
        raise PersistenceError(msg)
    return feature_lineage

def get_pipeline_cfg_hash(training_metadata: dict) -> str:
    pipeline_cfg_hash = training_metadata.get("config_fingerprint", {}).get("pipeline_cfg_hash")
    if not pipeline_cfg_hash:
        msg = "Training metadata is missing pipeline configuration hash information."
        logger.error(msg)
        raise PersistenceError(msg)
    return pipeline_cfg_hash

def get_training_conda_env_hash(train_run_dir: Path) -> str:
    training_runtime_file = train_run_dir / "runtime.json"
    training_runtime = load_json(training_runtime_file)
    training_conda_env_hash = training_runtime.get("environment", {}).get("conda_env_hash")
    if not training_conda_env_hash:
        msg = f"Training runtime information is missing conda environment hash. Runtime file: {training_runtime_file}"
        logger.error(msg)
        raise PersistenceError(msg)
    return training_conda_env_hash