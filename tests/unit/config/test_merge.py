"""Unit tests for configuration merge and overlay helpers."""

from pathlib import Path
from typing import Any

import pytest
from ml.config.merge import apply_env_overlay, deep_merge, resolve_extends
from ml.exceptions import ConfigError, PipelineContractError

pytestmark = pytest.mark.unit


def test_deep_merge_merges_nested_dicts_with_later_values_winning() -> None:
    """Deep-merge nested dictionaries while preserving earlier unrelated keys."""
    base = {
        "model": {
            "name": "catboost",
            "params": {"depth": 6, "learning_rate": 0.1},
        },
        "seed": 1,
    }
    override = {
        "model": {
            "params": {"depth": 8},
        },
        "feature": {"enabled": True},
    }

    merged = deep_merge([base, override])

    assert merged == {
        "model": {
            "name": "catboost",
            "params": {"depth": 8, "learning_rate": 0.1},
        },
        "seed": 1,
        "feature": {"enabled": True},
    }


def test_deep_merge_creates_independent_result_without_aliasing_inputs() -> None:
    """Return a deep-copied merged structure so later mutations stay isolated."""
    first = {"a": {"x": [1, 2]}}
    second = {"b": {"y": {"z": 3}}}

    merged = deep_merge([first, second])
    merged["a"]["x"].append(99)
    merged["b"]["y"]["z"] = 42

    assert first == {"a": {"x": [1, 2]}}
    assert second == {"b": {"y": {"z": 3}}}


def test_resolve_extends_merges_recursive_parents_relative_to_each_file(tmp_path: Path) -> None:
    """Resolve recursive parent configs and merge them in parent-to-child order."""
    common = tmp_path / "common.yaml"
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"

    common.write_text(
        "\n".join(
            [
                "pipeline:",
                "  retries: 1",
                "  timeout: 30",
                "model:",
                "  params:",
                "    depth: 6",
            ]
        ),
        encoding="utf-8",
    )
    parent.write_text(
        "\n".join(
            [
                "extends:",
                "  - common.yaml",
                "pipeline:",
                "  retries: 2",
                "model:",
                "  params:",
                "    learning_rate: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    child.write_text(
        "\n".join(
            [
                "extends:",
                "  - parent.yaml",
                "pipeline:",
                "  timeout: 45",
                "feature_flags:",
                "  enable_checks: true",
            ]
        ),
        encoding="utf-8",
    )

    cfg = resolve_extends(
        cfg={"extends": ["child.yaml"]},
        base_path=tmp_path,
    )

    assert cfg["pipeline"] == {"retries": 2, "timeout": 45}
    assert cfg["model"]["params"] == {"depth": 6, "learning_rate": 0.1}
    assert cfg["feature_flags"] == {"enable_checks": True}


def test_resolve_extends_raises_for_missing_parent_when_not_skipping(tmp_path: Path) -> None:
    """Raise contract error when referenced parent config is not found."""
    with pytest.raises(PipelineContractError, match="Extended config not found"):
        resolve_extends(
            cfg={"extends": ["does-not-exist.yaml"]},
            base_path=tmp_path,
            skip_missing=False,
        )


def test_apply_env_overlay_merges_nested_overrides_when_overlay_exists(tmp_path: Path) -> None:
    """Apply environment overlay with deep-merge semantics for nested objects."""
    env_file = tmp_path / "dev.yaml"
    env_file.write_text(
        "\n".join(
            [
                "pipeline:",
                "  retries: 5",
                "model:",
                "  params:",
                "    learning_rate: 0.05",
            ]
        ),
        encoding="utf-8",
    )
    base_cfg: dict[str, Any] = {
        "pipeline": {"retries": 1, "timeout": 30},
        "model": {"params": {"depth": 6}},
    }

    merged = apply_env_overlay(base_cfg, env="dev", env_path=env_file, skip_missing=False)

    assert merged == {
        "pipeline": {"retries": 5, "timeout": 30},
        "model": {"params": {"depth": 6, "learning_rate": 0.05}},
    }


def test_apply_env_overlay_requires_env_when_skip_missing_is_false(tmp_path: Path) -> None:
    """Fail fast if no env key is provided and skip-missing behavior is disabled."""
    with pytest.raises(ConfigError, match="Environment not specified"):
        apply_env_overlay({"a": 1}, env=None, env_path=tmp_path / "dev.yaml", skip_missing=False)
