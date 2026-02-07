import logging
import json
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

from ml.utils.git import get_git_commit
from ml.exceptions import PersistenceError
from ml.config.validation_schemas.model_cfg import SearchModelConfig

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)
def save_metadata(model_cfg: SearchModelConfig, search_results: dict, owner: str, *, experiment_id: str, timestamp: str) -> Path:
    problem = model_cfg.problem
    segment = model_cfg.segment.name
    version = model_cfg.version

    if experiment_id is None:
        experiment_id = f"{timestamp}_{uuid4().hex[:8]}"

    run_dir = EXPERIMENTS_DIR / problem / segment / version / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = model_cfg.meta
    sources = meta.sources if meta.sources else {}
    env = meta.env if meta.env else "default"
    best_params_path = meta.best_params_path if meta.best_params_path else "none"

    pipeline_version = model_cfg.pipeline.version if model_cfg.pipeline else "none"

    exp_path = run_dir / "experiment.json"

    git_commit = get_git_commit(Path("."))
    config_hash = meta.config_hash if meta.config_hash else "none"
    validation_status = meta.validation_status if meta.validation_status else "unknown"

    record = {
        "metadata": {
            "problem": problem,
            "segment": segment,
            "version": version,
            "experiment_id": experiment_id,
            "sources": sources,
            "env": env,
            "best_params_path": best_params_path,
            "algorithm": model_cfg.algorithm.value if model_cfg.algorithm else None,
            "pipeline_version": pipeline_version,
            "created_by": "search.py",
            "created_at": timestamp,
            "owner": owner,
            "feature_store": model_cfg.feature_store.model_dump() if model_cfg.feature_store else {},
            "seed": model_cfg.seed if model_cfg.seed is not None else "none",
            "hardware": model_cfg.search.hardware.model_dump() if model_cfg.search and model_cfg.search.hardware else {},
            "git_commit": git_commit,
            "config_hash": config_hash,
            "validation_status": validation_status,
        },
        "config": model_cfg.model_dump(by_alias=True),
        "search_results": search_results
    }

    try:
        with exp_path.open("w") as f:
            json.dump(record, f, indent=4, sort_keys=True, default=str)
        logger.info("Saved hyperparameter search experiment to %s", exp_path)
    except Exception:
        logger.exception("Failed to save experiment to %s", exp_path)
        raise PersistenceError(f"Failed to save experiment to {exp_path}")

    return exp_path
