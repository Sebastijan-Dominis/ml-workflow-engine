"""Tests for the training CLI and helper functions in ml.training.train_scripts.

These tests exercise argument parsing, configuration loading and
validation, and the main dispatch behavior for unsupported tasks
and algorithms.
"""

import sys
import yaml
import pytest

from pathlib import Path
from ml.training.train_scripts import train as train_module
from ml.training.train_scripts.train import (
    parse_args,
    load_config,
    validate_config_schema,
    main,
)

def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CLI argument parsing returns the expected namespace.

    This test simulates a minimal invocation by patching `sys.argv` and
    asserts the parsed `name_version` value is returned.
    """

    monkeypatch.setattr(sys, "argv", ["train.py", "--name_version", "m_v1"]) 
    args = parse_args()
    assert args.name_version == "m_v1"


def test_load_config_reads_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `load_config` reads configuration YAML from the train configs folder.

    A temporary `train_configs` layout is created and a minimal YAML
    file is written; the loader should return the expected dictionary.
    """
    # Create expected path structure under tmp
    base = tmp_path / "ml" / "training" / "train_configs"
    base.mkdir(parents=True)
    cfg_file = base / "m_v1.yaml"
    data = {"name": "m", "task": "binary"}
    cfg_file.write_text(yaml.safe_dump(data))

    # Run with cwd at tmp_path so relative path resolves
    monkeypatch.chdir(tmp_path)
    loaded = load_config("m_v1")
    assert loaded["name"] == "m"


def test_validate_config_schema_exits_on_invalid() -> None:
    """Ensure `validate_config_schema` raises SystemExit on invalid config."""
    # Pass an invalid config that lacks required keys
    bad_cfg = {"foo": "bar"}
    with pytest.raises(SystemExit):
        validate_config_schema(bad_cfg)

def test_unsupported_task_and_algorithm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert `main()` raises SystemExit for unsupported tasks and algorithms."""
    # Mock parse_args to return a dummy name_version
    monkeypatch.setattr(
        train_module,
        "parse_args",
        lambda: type("A", (), {"name_version": "dummy_model_v1"})()
    )

    # Mock load_config to return unsupported task
    monkeypatch.setattr(
        train_module,
        "load_config",
        lambda _: {"task": "unsupported_task", "model": {"algorithm": "catboost"}}
    )

    # Mock setup_logging to do nothing
    monkeypatch.setattr(train_module, "setup_logging", lambda: None)

    # Check that SystemExit is raised for unsupported task
    with pytest.raises(SystemExit):
        main()

    # Now test unsupported algorithm
    monkeypatch.setattr(
        train_module,
        "load_config",
        lambda _: {"task": "binary_classification", "model": {"algorithm": "unsupported_algo"}}
    )

    with pytest.raises(SystemExit):
        main()