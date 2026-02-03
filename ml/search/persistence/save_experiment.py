import logging
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)

from ml.utils.utils import get_git_commit

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

def save_experiment(model_cfg: dict, search_results: dict, owner: str) -> Path:
    """
    Persist hyperparameter search results in a unique run directory based on timestamp.

    Returns:
        Path to the saved experiment JSON file.
    """
    problem = model_cfg["problem"]
    segment = model_cfg["segment"]["name"]
    version = model_cfg["version"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{timestamp}_{uuid4().hex[:8]}"
    run_dir = EXPERIMENTS_DIR / problem / segment / version / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    sources = model_cfg.get("_meta", {}).get("sources", {})
    env = model_cfg.get("_meta", {}).get("env", "default")
    best_params_path = model_cfg.get("_meta", {}).get("best_params_path", "none")

    pipeline_version = model_cfg.get("pipeline", {}).get("version", "none")

    exp_path = run_dir / "experiment.json"

    git_commit = get_git_commit(Path("."))
    config_hash = model_cfg.get("_meta", {}).get("config_hash", "none")
    validation_status = model_cfg.get("_meta", {}).get("validation_status", "unknown")

    record = {
        "metadata": {
            "problem": problem,
            "segment": segment,
            "version": version,
            "experiment_id": experiment_id,
            "sources": sources,
            "env": env,
            "best_params_path": best_params_path,
            "algorithm": model_cfg.get("algorithm"),
            "pipeline_version": pipeline_version,
            "created_by": "search.py",
            "created_at": timestamp,
            "owner": owner,
            "feature_store": model_cfg.get("feature_store", {}),
            "seed": model_cfg.get("seed", "none"),
            "hardware": model_cfg.get("hardware", {}),
            "git_commit": git_commit,
            "config_hash": config_hash,
            "validation_status": validation_status,
        },
        "config": model_cfg,
        "search_results": search_results
    }

    try:
        with exp_path.open("w") as f:
            json.dump(record, f, indent=4, sort_keys=True)
        logger.info("Saved hyperparameter search experiment to %s", exp_path)
    except Exception:
        logger.exception("Failed to save experiment to %s", exp_path)
        raise

    return exp_path
