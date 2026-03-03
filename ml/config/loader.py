"""Configuration loading, composition, and validation entrypoints."""

import logging
from pathlib import Path
from typing import Any, Literal, overload

from ml.config.best_params import MergeTarget, apply_best_params
from ml.config.merge import apply_env_overlay, resolve_extends
from ml.config.validation import validate_model_config
from ml.config.schemas.model_cfg import (SearchModelConfig,
                                                    TrainModelConfig)
from ml.exceptions import ConfigError, UserError
from ml.utils.loaders import load_yaml

logger = logging.getLogger(__name__)

def load_config(
    path: Path,
    *,
    env: str = "default",
    cfg_type: Literal["search", "train"],
    search_dir: Path | None = None,
    merge_target: MergeTarget = "training",
    skip_missing_extends: bool = False,
    skip_missing_env: bool = True,
) -> dict[str, Any]:
    """Load and compose raw config with extends/env overlays and metadata.

    Args:
        path: Primary config file path.
        env: Environment overlay key.
        cfg_type: Configuration type (search or train).
        search_dir: Search output directory used by train configs.
        merge_target: Section where best params are merged for train configs.
        skip_missing_extends: Whether to ignore missing extended config files.
        skip_missing_env: Whether to ignore missing env overlay files.

    Returns:
        dict[str, Any]: Merged raw configuration dictionary.
    """
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

    base_env_path = Path("configs/env")
    env_path = (base_env_path / f"{env}.yaml").resolve()
    cfg = apply_env_overlay(cfg, env, env_path=env_path, skip_missing=skip_missing_env)

    cfg["_meta"]["env"] = env

    if cfg_type == "train":
        if search_dir is None:
            msg = "search_dir must be provided for training configs"
            logger.error(msg)
            raise UserError(msg)
        
        best_params_path = search_dir / "metadata.json"
        
        cfg = apply_best_params(cfg, best_params_path, merge_target=merge_target, strict=True)

        cfg["_meta"]["best_params_path"] = (
            str(best_params_path)
        )

    cfg["_meta"]["validation_status"] = "missing"

    logger.info("Final merged config: %s", cfg)
    
    return cfg

@overload
def load_and_validate_config(
    path: Path,
    search_dir: None = None,
    *,
    cfg_type: Literal["search"],
    env: str = "default",
) -> SearchModelConfig:
    """Typed overload for loading and validating search configurations.

    Args:
        path: Config file path.
        search_dir: Search directory placeholder (unused for search configs).
        cfg_type: Literal discriminator for search config.
        env: Environment overlay key.

    Returns:
        SearchModelConfig: Validated search configuration object.
    """

    ...

@overload
def load_and_validate_config(
    path: Path,
    search_dir: Path,
    *,
    cfg_type: Literal["train"],
    env: str = "default",
) -> TrainModelConfig:
    """Typed overload for loading and validating training configurations.

    Args:
        path: Config file path.
        search_dir: Search output directory used for best-params merge.
        cfg_type: Literal discriminator for train config.
        env: Environment overlay key.

    Returns:
        TrainModelConfig: Validated training configuration object.
    """

    ...

def load_and_validate_config(
    path: Path,
    search_dir: Path | None = None,
    *,
    cfg_type: Literal["search", "train"],
    env: str = "default",
) -> SearchModelConfig | TrainModelConfig:
    """Load and validate model config into typed schema objects.

    Args:
        path: Config file path.
        search_dir: Optional search run directory for training configs.
        cfg_type: Config type discriminator.
        env: Environment overlay key.

    Returns:
        SearchModelConfig | TrainModelConfig: Validated typed configuration.
    """

    cfg_raw = load_config(path, env=env, search_dir=search_dir, cfg_type=cfg_type)
    cfg = validate_model_config(cfg_raw, cfg_type=cfg_type)
    return cfg