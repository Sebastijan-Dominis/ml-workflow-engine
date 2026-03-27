import sys

import pipelines.post_promotion.monitor as monitor_mod
import pytest
from ml.types.latest import LatestSnapshot


def test_parse_args_success(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--problem", "no_show", "--segment", "segA"])
    args = monitor_mod.parse_args()

    assert args.problem == "no_show"
    assert args.segment == "segA"
    assert args.inference_run_id == LatestSnapshot.LATEST.value
    assert args.logging_level == "INFO"


def test_parse_args_missing_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--problem", "no_show"])  # missing segment
    with pytest.raises(SystemExit) as exc:
        monitor_mod.parse_args()
    assert exc.value.code == 2
