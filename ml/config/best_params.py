import logging
logger = logging.getLogger(__name__)
from typing import Any, Literal
import json
from pathlib import Path

from ml.config.merge import deep_merge
from ml.exceptions import ConfigError

MergeTarget = Literal["training", "model", "ensemble"]

def apply_best_params(
    cfg: dict[str, Any],
    best_params_path: Path | None,
    *,
    merge_target: MergeTarget = "training",
    strict: bool = True,
) -> dict[str, Any]:
    if not best_params_path:
        return cfg  

    if not best_params_path.exists():
        msg = f"best_params file not found: {best_params_path}"
        if strict:
            logger.error(msg)
            raise ConfigError(msg)
        logger.warning(msg)
        return cfg

    try:
        with best_params_path.open("r") as f:
            experiment_data = json.load(f)

        best_params = (
            experiment_data
            .get("search_results", {})
            .get("best_params", {})
        )

        if not best_params:
            if strict:
                msg = f"No best_params found in {best_params_path}"
                logger.error(msg)
                raise ConfigError(msg)
            logger.warning("No best_params found in %s", best_params_path)
            return cfg

        logger.info("Applied best_params from %s", best_params_path)
        return deep_merge([cfg, {merge_target: best_params}])

    except Exception:
        if strict:
            logger.exception("Error applying best_params from %s", best_params_path)
            raise
        logger.warning("Failed to apply best_params from %s", best_params_path)
        return cfg