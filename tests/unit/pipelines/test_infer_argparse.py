import sys

import pipelines.post_promotion.infer as infer_mod
import pytest


def test_parse_args_success(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--problem", "no_show", "--segment", "segA", "--snapshot-bindings-id", "sb1"])
    args = infer_mod.parse_args()

    assert args.problem == "no_show"
    assert args.segment == "segA"
    assert args.snapshot_bindings_id == "sb1"
    assert args.logging_level == "INFO"


def test_parse_args_missing_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--problem", "no_show", "--segment", "segA"])  # missing snapshot-bindings-id
    with pytest.raises(SystemExit) as exc:
        infer_mod.parse_args()
    assert exc.value.code == 2
