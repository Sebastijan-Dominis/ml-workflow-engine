import json
import logging
from pathlib import Path
from typing import Any, Literal

from ml.config.merge import deep_merge
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

MergeTarget = Literal["training", "model", "ensemble"]

MODEL_KEYS = {"depth", "learning_rate", "l2_leaf_reg", "random_strength", "min_data_in_leaf", "colsample_bylevel", "border_count"}
ENSEMBLE_KEYS = {"bagging_temperature"}

def unflatten_best_params(flat: dict[str, Any]) -> dict[str, Any]:
    model: dict[str, Any] = {}
    ensemble: dict[str, Any] = {}
    other: dict[str, Any] = {}

    for k, v in flat.items():
        key_name = k.split("__")[-1] if "__" in k else k
        if key_name in MODEL_KEYS:
            model[key_name] = v
        elif key_name in ENSEMBLE_KEYS:
            ensemble[key_name] = v
        else:
            other[k] = v  # leave iterations, early_stopping_rounds, etc.

    result = {**other}
    if model:
        result["model"] = {**result.get("model", {}), **model}
    if ensemble:
        result["ensemble"] = {**result.get("ensemble", {}), **ensemble}
    return result

def apply_best_params(
    cfg: dict[str, Any],
    best_params_path: Path,
    *,
    merge_target: MergeTarget = "training",
    strict: bool = True,
) -> dict[str, Any]:
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
            .get("best_model_params", {})
        )

        if not best_params:
            if strict:
                msg = f"No best_params found in {best_params_path}"
                logger.error(msg)
                raise ConfigError(msg)
            logger.warning("No best_params found in %s", best_params_path)
            return cfg

        logger.debug("Applied best_params from %s", best_params_path)

        best_params_structured = unflatten_best_params(best_params)
        logger.debug("Structured best_params: %s", best_params_structured)

        return deep_merge([cfg, {merge_target: best_params_structured}])

    except Exception:
        if strict:
            logger.exception("Error applying best_params from %s", best_params_path)
            raise
        logger.warning("Failed to apply best_params from %s", best_params_path)
        return cfg