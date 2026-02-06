"""Tests for the training CLI and helper functions in ml.training.train_scripts.

These tests exercise argument parsing, configuration loading, and the
main dispatch behavior for unsupported algorithms.
"""

import sys
import yaml
import pytest

from pathlib import Path
from ml.training.train_scripts import train as train_module
from ml.training.train_scripts.train import (
    parse_args,
    load_train_configs,
    main,
)

def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CLI argument parsing returns the expected namespace."""

    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "--problem", "cancellation", "--segment", "global",
         "--version", "v1", "--experiment-id", "20260101_000000_abc12345"],
    )
    args = parse_args()
    assert args.problem == "cancellation"
    assert args.segment == "global"
    assert args.version == "v1"
    assert args.experiment_id == "20260101_000000_abc12345"


def test_load_train_configs_reads_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `load_train_configs` reads the expected YAML file."""
    base = tmp_path / "configs" / "train" / "cancellation" / "global"
    base.mkdir(parents=True)
    cfg_file = base / "v1.yaml"
    data = {"name": "m", "task": "binary"}
    cfg_file.write_text(yaml.safe_dump(data))

    monkeypatch.chdir(tmp_path)
    loaded = load_train_configs("cancellation", "global", "v1")
    assert loaded["name"] == "m"


def test_unsupported_algorithm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Assert `main()` returns a non-zero exit code for unsupported algorithms."""
    monkeypatch.setattr(
        train_module,
        "parse_args",
        lambda: type("A", (), {
            "problem": "cancellation",
            "segment": "global",
            "version": "v1",
            "experiment_id": "20260101_000000_abc12345",
            "logging_level": "INFO",
        })(),
    )

    monkeypatch.setattr(
        train_module,
        "load_model_specs",
        lambda p, s, v, log: {"algorithm": "unsupported_algo"},
    )
    monkeypatch.setattr(
        train_module,
        "validate_model_specs",
        lambda raw, log: raw,
    )
    monkeypatch.setattr(
        train_module,
        "load_train_configs",
        lambda p, s, v: {},
    )

    # Mock setup_logging to do nothing
    monkeypatch.setattr(train_module, "setup_logging", lambda *a, **kw: None)

    result = main()
    assert result != 0