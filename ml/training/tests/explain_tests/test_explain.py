"""Tests for the explain CLI and model configuration helpers.

These unit tests validate argument parsing, model config lookup, and
dispatch behavior for the explain entry point.
"""

import yaml
import pytest

from pathlib import Path

from ml.training.explain_scripts import explain

# Commenting out imports for possible future use
from ml.training.explain_scripts.explain import (
    parse_args,
    get_model_configs,
    # explain_catboost,
)

def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CLI argument parsing for the explain entry point works."""

    monkeypatch.setattr("sys.argv", ["explain.py", "--name_and_version", "dummy_model_v1"])
    args = parse_args()
    assert args.name_and_version == "dummy_model_v1"

def test_get_model_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dummy_models_config) -> None:
    """Verify model configuration lookup reads `configs/models.yaml` correctly."""

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(yaml.safe_dump(dummy_models_config))

    monkeypatch.chdir(tmp_path)

    out = get_model_configs("dummy_model_v1")
    assert out["algorithm"] == "catboost"

def test_main_dispatches_catboost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert `main()` dispatches to CatBoost explainability helper when configured."""

    from ml.training.explain_scripts import explain

    monkeypatch.setattr(
        explain,
        "parse_args",
        lambda: type("A", (), {"name_and_version": "dummy_model_v1"})()
    )

    monkeypatch.setattr(
        explain,
        "get_model_configs",
        lambda _: {"algorithm": "catboost"}
    )

    called = {}
    monkeypatch.setattr(
        explain,
        "explain_catboost",
        lambda cfg: called.setdefault("ok", True)
    )

    monkeypatch.setattr(explain, "setup_logging", lambda: None)

    explain.main()
    assert called["ok"] is True

def test_main_unsupported_algorithm(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the algorithm is unsupported, `main()` should raise ValueError."""

    monkeypatch.setattr(
        explain,
        "parse_args",
        lambda: type("A", (), {"name_and_version": "dummy_model_v1"})()
    )
    monkeypatch.setattr(
        explain,
        "get_model_configs",
        lambda _: {"algorithm": "fake_algorithm"}
    )
    monkeypatch.setattr(explain, "setup_logging", lambda: None)

    with pytest.raises(ValueError, match="Unsupported algorithm"):
        explain.main()