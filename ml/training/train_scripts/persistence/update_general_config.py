"""Utilities to update the project's global models registry.

The training pipeline writes per-model metadata and artifacts. This
module exposes ``update_general_config`` which appends a new immutable
training run to the central ``configs/models.yaml`` registry and updates
the `latest` pointer for downstream consumers.
"""

import logging
from datetime import datetime
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def update_general_config(cfg: dict, best_threshold: float | None = None) -> str:
    """Append a new training run to ``configs/model_registry/models.yaml``.

    Each training invocation creates a new immutable run entry. Historical
    runs are never overwritten. A `latest` pointer is updated to reference
    the newest run.

    Args:
        cfg (dict): Validated configuration dictionary describing the model.
        best_threshold (float | None): The best threshold value for the model, if applicable.
    Returns:
        str: The generated run_id for downstream use (evaluation, explainability).
    """

    model_key = f"{cfg['problem']}_{cfg['segment']['name']}_{cfg['version']}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_key}_run_{timestamp}"

    artifact_path = Path(
        f"ml/artifacts/{cfg['problem']}/{cfg['segment']['name']}/{cfg['version']}/{run_id}"
    )
    artifact_path.mkdir(parents=True, exist_ok=True)

    trained_on = datetime.now().strftime("%Y-%m-%d")

    # Immutable per-run entry
    run_entry = {
        "run_id": run_id,
        "trained_on": trained_on,
        "threshold": best_threshold,
        "artifacts": {
            "model": str(artifact_path / "model.joblib"),
            "pipeline": str(artifact_path / "pipeline.joblib"),
            "metadata": str(artifact_path / "metadata.json"),
            # These are expected to be populated later
            "metrics": None,
            "explainability": None,
        },
    }

    registry_path = Path("configs/model_registry/models.yaml")
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if registry_path.exists():
        with open(registry_path) as f:
            registry = yaml.safe_load(f) or {}
    else:
        registry = {}

    # Initialize model entry if it doesn't exist
    registry.setdefault(
        model_key,
        {
            "model_specs": f"configs/model_specs/{cfg['problem']}/{cfg['segment']['name']}/{cfg['version']}.yaml",
            "runs": {},
            "latest": None,
            "production": None,
        },
    )

    # Append new immutable run
    registry[model_key]["runs"][run_id] = run_entry
    registry[model_key]["latest"] = run_id

    with open(registry_path, "w") as f:
        yaml.safe_dump(registry, f, sort_keys=False)

    logger.info(
        "Registered new training run %s for model %s (set as latest)",
        run_id,
        model_key,
    )

    return run_id
