"""Config composition helpers for recursive merges and overlays."""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from ml.exceptions import ConfigError, PipelineContractError
from ml.utils.loaders import load_yaml

logger = logging.getLogger(__name__)

def deep_merge(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Recursively merge config dictionaries where later entries take precedence.

    Args:
        dicts: Ordered dictionaries to merge.

    Returns:
        dict[str, Any]: Deep-merged configuration dictionary.
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
    """Resolve recursive ``extends`` references and deep-merge parent configs.

    Args:
        cfg: Raw configuration dictionary.
        base_path: Directory used to resolve relative parent paths.
        skip_missing: Whether missing parent configs should be ignored.

    Returns:
        dict[str, Any]: Configuration with all parent configs merged.
    """

    parents = []

    for parent in cfg.get("extends", []):
        if not isinstance(parent, str):
            msg = f"Invalid extends entry: {parent}. Must be a string."
            logger.error(msg)
            raise ConfigError(msg)

        parent_path = (base_path / parent).resolve()

        if not parent_path.exists():
            if skip_missing:
                logger.warning("Skipped missing parent config: %s", parent_path)
                continue
            raise PipelineContractError(f"Extended config not found: {parent_path}")

        parent_cfg = load_yaml(parent_path)
        parent_cfg = resolve_extends(
            parent_cfg,
            parent_path.parent,
            skip_missing=skip_missing,
        )
        parents.append(parent_cfg)

    return deep_merge(parents + [cfg])

def apply_env_overlay(cfg: dict[str, Any], env: str | None, env_path: Path, skip_missing: bool = True) -> dict[str, Any]:
    """Apply environment overlay config on top of the base configuration.

    Args:
        cfg: Base configuration dictionary.
        env: Environment overlay key.
        env_path: Environment config file path.
        skip_missing: Whether missing overlays should be ignored.

    Returns:
        dict[str, Any]: Configuration after applying env overlay.
    """

    if not env:
        if skip_missing:
            logger.warning("No environment specified; skipping env overlay.")
            return cfg
        else:
            msg = "Environment not specified for env overlay."
            logger.error(msg)
            raise ConfigError(msg)

    if not env_path.exists():
        if skip_missing:
            logger.warning("Environment overlay %s not found in env_path: %s", env, env_path)
            return cfg
        else:
            msg = f"Environment overlay not found: {env_path}"
            logger.error(msg)
            raise PipelineContractError(msg)

    env_cfg = load_yaml(env_path)

    result = deep_merge([cfg, env_cfg])
    logger.debug("Applied environment overlay: %s", env)
    return result