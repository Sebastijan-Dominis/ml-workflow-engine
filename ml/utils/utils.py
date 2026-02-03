# TODO: Make more modular, split into multiple files.
import logging
logger = logging.getLogger(__name__)

import yaml
import subprocess
import hashlib
import json
import copy
from pathlib import Path
from pydantic_core import ValidationError
from copy import deepcopy
from typing import Any, Literal, Dict

MergeTarget = Literal["training", "model", "ensemble"]

from ml.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig

def get_git_commit(repo_dir: Path = Path(".")) -> str:
    try:
        # Find the top-level git directory
        top_level = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        # Get the HEAD commit hash
        commit_hash = subprocess.check_output(
            ["git", "-C", top_level, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        return commit_hash
    except subprocess.CalledProcessError:
        return "unknown"

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")

    return cfg

def deep_merge(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Iteratively merge a list of dictionaries. Later dicts override earlier ones.
    Only copies nested dicts, not primitives, for efficiency.
    """
    result: dict[str, Any] = {}

    for d in dicts:
        stack: list[tuple[dict[str, Any], dict[str, Any]]] = [(result, d)]
        while stack:
            res_sub, d_sub = stack.pop()
            for k, v in d_sub.items():
                if (
                    k in res_sub
                    and isinstance(res_sub[k], dict)
                    and isinstance(v, dict)
                ):
                    stack.append((res_sub[k], v))
                else:
                    res_sub[k] = deepcopy(v)
    return result

def resolve_extends(
    cfg: dict[str, Any],
    base_path: Path,
    *,
    skip_missing: bool = False,
) -> dict[str, Any]:
    parents = []

    for parent in cfg.get("extends", []):
        if not isinstance(parent, str):
            msg = f"Invalid extends entry: {parent}. Must be a string."
            logger.error(msg)
            raise ValueError(msg)

        parent_path = (base_path / parent).resolve()

        if not parent_path.exists():
            if skip_missing:
                logger.warning("Skipped missing parent config: %s", parent_path)
                continue
            raise FileNotFoundError(f"Extended config not found: {parent_path}")

        parent_cfg = load_yaml(parent_path)
        parent_cfg = resolve_extends(
            parent_cfg,
            parent_path.parent,
            skip_missing=skip_missing,
        )
        parents.append(parent_cfg)

    return deep_merge(parents + [cfg])

def apply_env_overlay(cfg: dict[str, Any], env: str | None, base_path: Path, skip_missing: bool = True) -> dict[str, Any]:
    if not env:
        if skip_missing:
            logger.warning("No environment specified; skipping env overlay.")
            return cfg
        else:
            msg = "Environment not specified for env overlay."
            logger.error(msg)
            raise ValueError(msg)

    env_path = (base_path / "env" / f"{env}.yaml").resolve()

    if not env_path.exists():
        if skip_missing:
            logger.warning("Environment overlay not found: %s", env)
            return cfg
        else:
            msg = f"Environment overlay not found: {env_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

    env_cfg = load_yaml(env_path)
    logger.info("Applied environment overlay: %s", env)

    return deep_merge([cfg, env_cfg])

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
            raise FileNotFoundError(msg)
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
                raise ValueError(msg)
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

def compute_config_hash(cfg: dict[str, Any]) -> str:
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.pop("_meta", None)  # remove infrastructure-only metadata
    payload = json.dumps(cfg_copy, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def load_config(
    path: Path,
    *,
    env: str = "default",
    best_params_path: Path | None = None,
    merge_target: MergeTarget = "training",
    skip_missing_extends: bool = False,
    skip_missing_env: bool = True,
) -> dict:
    cfg = load_yaml(path)

    cfg = resolve_extends(
        cfg,
        base_path=path.parent,
        skip_missing=skip_missing_extends,
    )

    cfg.setdefault("_meta", {})["sources"] = {
        "main": str(path),
        "extends": cfg.get("extends", []),
    }

    cfg = apply_env_overlay(cfg, env, base_path=path.parent, skip_missing=skip_missing_env)

    cfg["_meta"]["env"] = env

    cfg = apply_best_params(cfg, best_params_path, merge_target=merge_target, strict=True)

    cfg["_meta"]["best_params_path"] = (
        str(best_params_path) if best_params_path else "none"
    )

    cfg["_meta"]["validation_status"] = "missing"

    cfg["_meta"]["config_hash"] = compute_config_hash(cfg)

    logger.debug("Final merged config: %s", cfg)
    
    return cfg

def validate_model_config(cfg_raw: Dict[str, Any], cfg_type: Literal["search", "train"]) -> Dict[str, Any]:
    """
    Validate a raw model config dict using the appropriate Pydantic schema.

    Args:
        cfg_raw (Dict[str, Any]): Raw config loaded from YAML/JSON.
        cfg_type (Literal["search", "train"]): Type of config to validate.

    Returns:
        Dict[str, Any]: Validated config as a dictionary.

    Raises:
        ValueError: If cfg_type is unknown or validation fails.
    """
    cfg_raw.setdefault("_meta", {})
    try:
        if cfg_type == "search":
            validated_cfg = SearchModelConfig(**cfg_raw)
        elif cfg_type == "train":
            validated_cfg = TrainModelConfig(**cfg_raw)
        else:
            cfg_raw["_meta"]["validation_status"] = "failed"
            msg = f"Unknown config type: {cfg_type}"
            logger.error(msg)
            raise ValueError(msg)

        cfg_raw["_meta"]["validation_status"] = "ok"
        cfg_raw["_meta"].pop("validation_errors", None)  # Clear previous errors if any
        return validated_cfg.model_dump()

    except ValidationError as e:
        cfg_raw["_meta"]["validation_status"] = "failed"
        logger.error("Model config validation failed for type '%s':", cfg_type)
        for err in e.errors():
            field_path = ".".join(map(str, err.get("loc", [])))
            msg = err.get("msg", "Unknown error")
            logger.error(" - Field '%s': %s", field_path, msg)

        if isinstance(cfg_raw, dict):
            cfg_raw["_meta"]["validation_errors"] = e.errors()

        raise ValueError(f"Validation failed for {cfg_type} config") from e

def load_and_validate_config(
        path: Path,
        *,
        cfg_type: Literal["search", "train"],
        env: str = "default",
    ) -> dict:

    cfg_raw = load_config(path, env=env)
    cfg = validate_model_config(cfg_raw, cfg_type=cfg_type)
    cfg["_meta"] = {**cfg.get("_meta", {}), **cfg_raw.get("_meta", {})}
    return cfg