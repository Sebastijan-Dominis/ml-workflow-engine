import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from ml.utils.loaders import load_yaml

logger = logging.getLogger(__name__)

def deep_merge(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Uses manual recursive merging to merge a list of dicts into one. 
    Later dicts take precedence over earlier ones.
    Primitives and lists are replaced.
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