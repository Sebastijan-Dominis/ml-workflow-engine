import json
import logging
from pathlib import Path

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)

def save_metrics(metrics: dict[str, float], *, train_run_id: str, experiment_dir: Path) -> None:
    metrics_file = experiment_dir / "training" / train_run_id / "metrics.json"

    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics successfully saved to {metrics_file}.")
    except Exception as e:
        msg = f"Failed to save metrics to {metrics_file}"
        logger.exception(msg)
        raise PersistenceError(msg) from e