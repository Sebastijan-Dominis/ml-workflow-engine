import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

from ml.utils import get_git_commit

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

def save_experiment(model_specs: dict, search_cfg: dict, search_results: dict) -> Path:
    """
    Persist hyperparameter search results in a unique run directory based on timestamp.

    Returns:
        Path to the saved experiment JSON file.
    """
    problem = model_specs["problem"]
    segment = model_specs["segment"]["name"]
    version = model_specs["version"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EXPERIMENTS_DIR / problem / segment / version / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    exp_path = run_dir / "experiment.json"

    git_commit = get_git_commit(Path("."))

    record = {
        "metadata": {
            "problem": problem,
            "segment": segment,
            "version": version,
            "algorithm": model_specs.get("algorithm"),
            "pipeline": model_specs.get("pipeline"),
            "created_at": timestamp,
            "feature_store": model_specs.get("feature_store", {}),
            "seed": search_cfg.get("seed", None),
            "hardware": search_cfg.get("hardware", None),
            "git_commit": git_commit,
        },
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
