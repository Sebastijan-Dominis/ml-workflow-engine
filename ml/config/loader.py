import logging
from pathlib import Path
from typing import Any, Literal, overload

from ml.config.best_params import MergeTarget, apply_best_params
from ml.config.hashing import compute_config_hash
from ml.config.merge import apply_env_overlay, resolve_extends
from ml.config.validation import validate_model_config
from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import ConfigError, UserError
from ml.utils.loaders import load_yaml

logger = logging.getLogger(__name__)

def load_config(
    path: Path,
    *,
    env: str = "default",
    cfg_type: Literal["search", "train"],
    experiment_dir: Path | None = None,
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

    if cfg_type == "train":
        if experiment_dir is None:
            msg = "experiment_dir must be provided for training configs"
            logger.error(msg)
            raise UserError(msg)
        
        best_params_path = experiment_dir / "experiment.json"
        
        cfg = apply_best_params(cfg, best_params_path, merge_target=merge_target, strict=True)

        cfg["_meta"]["best_params_path"] = (
            str(best_params_path)
        )

    cfg["_meta"]["validation_status"] = "missing"

    logger.debug("Final merged config: %s", cfg)
    
    return cfg

@overload
def load_and_validate_config(
    path: Path,
    experiment_dir: None = None,
    *,
    cfg_type: Literal["search"],
    env: str = "default",
) -> SearchModelConfig: ...

@overload
def load_and_validate_config(
    path: Path,
    experiment_dir: Path,
    *,
    cfg_type: Literal["train"],
    env: str = "default",
) -> TrainModelConfig: ...

def load_and_validate_config(
    path: Path,
    experiment_dir: Path | None = None,
    *,
    cfg_type: Literal["search", "train"],
    env: str = "default",
) -> SearchModelConfig | TrainModelConfig:
    cfg_raw = load_config(path, env=env, experiment_dir=experiment_dir, cfg_type=cfg_type)
    cfg = validate_model_config(cfg_raw, cfg_type=cfg_type)
    return cfg