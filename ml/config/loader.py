import logging
logger = logging.getLogger(__name__)
import yaml
from pathlib import Path
from typing import Any, Literal, overload

from ml.utils.loader import load_yaml
from ml.config.best_params import MergeTarget, apply_best_params
from ml.config.merge import resolve_extends, apply_env_overlay
from ml.config.hashing import compute_config_hash
from ml.config.validation import validate_model_config
from ml.exceptions import ConfigError
from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig

def load_config(
    path: Path,
    *,
    env: str = "default",
    best_params_path: Path | None = None,
    merge_target: MergeTarget = "training",
    skip_missing_extends: bool = False,
    skip_missing_env: bool = True,
) -> dict[str, Any]:
    cfg = load_yaml(path)

    try:
        cfg = resolve_extends(
            cfg,
            base_path=path.parent,
            skip_missing=skip_missing_extends,
        )
    except FileNotFoundError as e:
        msg = f"Extended config not found in {path}: {e}"
        logger.error(msg)
        raise ConfigError(msg) from e
    except ValueError as e:
        msg = f"Invalid extends entry in {path}: {e}"
        logger.error(msg)
        raise ConfigError(msg) from e

    cfg.setdefault("_meta", {})["sources"] = {
        "main": str(path),
        "extends": cfg.get("extends", []),
    }
    cfg.pop("extends", None)  # merge directive, not model content
    try:
        cfg = apply_env_overlay(cfg, env, base_path=path.parent, skip_missing=skip_missing_env)
    except FileNotFoundError as e:
        msg = f"Environment overlay not found in {path}: {e}"
        logger.error(msg)
        raise ConfigError(msg) from e
    except ValueError as e:
        msg = f"Invalid env overlay in {path}: {e}"
        logger.error(msg)
        raise ConfigError(msg) from e

    cfg["_meta"]["env"] = env

    cfg = apply_best_params(cfg, best_params_path, merge_target=merge_target, strict=True)

    cfg["_meta"]["best_params_path"] = (
        str(best_params_path) if best_params_path else "none"
    )

    cfg["_meta"]["validation_status"] = "missing"

    cfg["_meta"]["config_hash"] = compute_config_hash(cfg)

    logger.debug("Final merged config: %s", cfg)
    
    return cfg

@overload
def load_and_validate_config(
    path: Path,
    *,
    cfg_type: Literal["search"],
    env: str = "default",
) -> SearchModelConfig: ...
@overload
def load_and_validate_config(
    path: Path,
    *,
    cfg_type: Literal["train"],
    env: str = "default",
) -> TrainModelConfig: ...

def load_and_validate_config(
    path: Path,
    *,
    cfg_type: Literal["search", "train"],
    env: str = "default",
) -> SearchModelConfig | TrainModelConfig:
    cfg_raw = load_config(path, env=env)
    cfg = validate_model_config(cfg_raw, cfg_type=cfg_type)
    return cfg