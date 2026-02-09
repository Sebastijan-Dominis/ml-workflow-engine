"""Tests for the explain CLI and model configuration helpers.

These unit tests validate argument parsing, model config lookup, and
dispatch behavior for the explain entry point.
"""

from pathlib import Path

import pytest
import yaml

from ml.runners.explainability import explain

# Commenting out imports for possible future use
from ml.runners.explainability.explain import (  # explain_catboost,
    get_model_configs,
    parse_args,
)


def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CLI argument parsing for the explain entry point works."""

    monkeypatch.setattr(
        "sys.argv",
        ["explain.py", "--problem", "cancellation", "--segment", "global",
         "--version", "v1", "--experiment-id", "20260101_000000_abc12345"],
    )
    args = parse_args()
    assert args.problem == "cancellation"
    assert args.segment == "global"
    assert args.version == "v1"
    assert args.experiment_id == "20260101_000000_abc12345"

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

    from ml.runners.explainability import explain

    monkeypatch.setattr(
        explain,
        "parse_args",
        lambda: type("A", (), {
            "problem": "cancellation",
            "segment": "global",
            "version": "v1",
            "experiment_id": "20260101_000000_abc12345",
            "logging_level": "INFO",
        })()
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

    monkeypatch.setattr(explain, "setup_logging", lambda *a, **kw: None)

    explain.main()
    assert called["ok"] is True

def test_main_unsupported_algorithm(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the algorithm is unsupported, `main()` should return non-zero."""

    monkeypatch.setattr(
        explain,
        "parse_args",
        lambda: type("A", (), {
            "problem": "cancellation",
            "segment": "global",
            "version": "v1",
            "experiment_id": "20260101_000000_abc12345",
            "logging_level": "INFO",
        })()
    )
    monkeypatch.setattr(
        explain,
        "get_model_configs",
        lambda _: {"algorithm": "fake_algorithm"}
    )
    monkeypatch.setattr(explain, "setup_logging", lambda *a, **kw: None)

    result = explain.main()
    assert result != 0