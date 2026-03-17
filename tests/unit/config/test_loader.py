"""Unit tests for configuration loading and validation entrypoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ml.config import loader
from ml.exceptions import ConfigError, UserError

pytestmark = pytest.mark.unit


def test_load_config_sets_meta_sources_and_clears_extends_before_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Populate source metadata and strip `extends` before env overlay is applied."""
    cfg_path = tmp_path / "train.yaml"
    captured: dict[str, Any] = {}

    monkeypatch.setattr(loader, "load_yaml", lambda _path: {"extends": ["base.yaml"], "model": {"a": 1}})
    monkeypatch.setattr(
        loader,
        "resolve_extends",
        lambda cfg, base_path, skip_missing: cfg,
    )

    def _fake_apply_env_overlay(
        cfg: dict[str, Any],
        env: str,
        *,
        env_path: Path,
        skip_missing: bool,
    ) -> dict[str, Any]:
        captured["cfg"] = cfg
        captured["env"] = env
        captured["env_path"] = env_path
        captured["skip_missing"] = skip_missing
        return {**cfg, "overlay": True}

    monkeypatch.setattr(loader, "apply_env_overlay", _fake_apply_env_overlay)

    out = loader.load_config(
        cfg_path,
        cfg_type="search",
        env="dev",
        skip_missing_env=False,
    )

    assert out["overlay"] is True
    assert out["_meta"]["validation_status"] == "missing"
    assert out["_meta"]["env"] == "dev"
    assert out["_meta"]["sources"] == {"main": Path(cfg_path).as_posix(), "extends": ["base.yaml"]}
    assert "extends" not in captured["cfg"]
    assert captured["env"] == "dev"
    assert captured["env_path"].name == "dev.yaml"
    assert captured["skip_missing"] is False


@pytest.mark.parametrize(
    ("error", "msg"),
    [
        (FileNotFoundError("missing parent"), "Extended config not found"),
        (ValueError("bad extends type"), "Invalid extends entry"),
    ],
)
def test_load_config_wraps_resolve_extends_errors_as_config_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    msg: str,
) -> None:
    """Normalize extend-resolution failures into `ConfigError` with stable messaging."""
    cfg_path = tmp_path / "search.yaml"

    monkeypatch.setattr(loader, "load_yaml", lambda _path: {"model": {"name": "catboost"}})

    def _raise(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(loader, "resolve_extends", _raise)

    with pytest.raises(ConfigError, match=msg):
        loader.load_config(cfg_path, cfg_type="search")


def test_load_config_train_requires_search_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject train config loading without a search output directory for best params."""
    cfg_path = tmp_path / "train.yaml"

    monkeypatch.setattr(loader, "load_yaml", lambda _path: {"training": {"iterations": 100}})
    monkeypatch.setattr(loader, "resolve_extends", lambda cfg, base_path, skip_missing: cfg)
    monkeypatch.setattr(loader, "apply_env_overlay", lambda cfg, env, *, env_path, skip_missing: cfg)

    with pytest.raises(UserError, match="search_dir must be provided for training configs"):
        loader.load_config(cfg_path, cfg_type="train", search_dir=None)


def test_load_config_train_applies_best_params_and_sets_meta_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Apply persisted best params for train configs and record metadata path."""
    cfg_path = tmp_path / "train.yaml"
    search_dir = tmp_path / "search" / "run_001"
    captured: dict[str, Any] = {}

    monkeypatch.setattr(loader, "load_yaml", lambda _path: {"training": {"iterations": 200}})
    monkeypatch.setattr(loader, "resolve_extends", lambda cfg, base_path, skip_missing: cfg)
    monkeypatch.setattr(loader, "apply_env_overlay", lambda cfg, env, *, env_path, skip_missing: cfg)

    def _fake_apply_best_params(
        cfg: dict[str, Any],
        best_params_path: Path,
        *,
        merge_target: str,
        strict: bool,
    ) -> dict[str, Any]:
        captured["best_params_path"] = best_params_path
        captured["merge_target"] = merge_target
        captured["strict"] = strict
        return {**cfg, "training": {**cfg.get("training", {}), "model": {"depth": 8}}}

    monkeypatch.setattr(loader, "apply_best_params", _fake_apply_best_params)

    out = loader.load_config(
        cfg_path,
        cfg_type="train",
        env="test",
        search_dir=search_dir,
    )

    assert out["training"]["model"] == {"depth": 8}
    assert captured["best_params_path"] == search_dir / "metadata.json"
    assert captured["merge_target"] == "training"
    assert captured["strict"] is True
    assert out["_meta"]["best_params_path"] == str(search_dir / "metadata.json")
    assert out["_meta"]["env"] == "test"


def test_load_and_validate_config_passes_through_loader_and_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Delegate raw-load and schema-validation to their dedicated collaborators."""
    cfg_path = tmp_path / "search.yaml"
    captured: dict[str, Any] = {}
    validated = SimpleNamespace(valid=True)

    def _fake_load_config(
        path: Path,
        *,
        env: str,
        cfg_type: str,
        search_dir: Path | None,
    ) -> dict[str, Any]:
        captured["path"] = path
        captured["env"] = env
        captured["cfg_type"] = cfg_type
        captured["search_dir"] = search_dir
        return {"pipeline": {"steps": []}}

    def _fake_validate_model_config(raw: dict[str, Any], *, cfg_type: str) -> object:
        captured["raw"] = raw
        captured["validated_cfg_type"] = cfg_type
        return validated

    monkeypatch.setattr(loader, "load_config", _fake_load_config)
    monkeypatch.setattr(loader, "validate_model_config", _fake_validate_model_config)

    result = loader.load_and_validate_config(
        cfg_path,
        cfg_type="search",
        env="prod",
    )

    assert result is validated
    assert captured["path"] == cfg_path
    assert captured["env"] == "prod"
    assert captured["cfg_type"] == "search"
    assert captured["search_dir"] is None
    assert captured["raw"] == {"pipeline": {"steps": []}}
    assert captured["validated_cfg_type"] == "search"
