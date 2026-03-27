import argparse
import sys

import pipelines.features.freeze as freeze_mod
import pytest


def test_parse_args_success(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--feature-set", "base_features", "--version", "v1"])
    args = freeze_mod.parse_args()

    assert args.feature_set == "base_features"
    assert args.version == "v1"
    assert args.snapshot_binding_key is None
    assert args.owner == "Sebastijan"
    assert args.logging_level == "INFO"


def test_parse_args_missing_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--feature-set", "base_features"])  # missing version
    with pytest.raises(SystemExit) as exc:
        freeze_mod.parse_args()
    assert exc.value.code == 2


def test_main_load_failure_returns_resolved_code(monkeypatch):
    args = argparse.Namespace(feature_set="fs", version="v1", snapshot_binding_key=None, owner="me", logging_level="INFO")
    monkeypatch.setattr(freeze_mod, "parse_args", lambda: args)
    monkeypatch.setattr(freeze_mod, "bootstrap_logging", lambda *a, **k: None)

    def raise_load(fs, v):
        raise RuntimeError("boom")

    monkeypatch.setattr(freeze_mod, "load_feature_registry", raise_load)
    monkeypatch.setattr(freeze_mod, "resolve_exit_code", lambda e: 99)

    res = freeze_mod.main()

    assert res == 99
