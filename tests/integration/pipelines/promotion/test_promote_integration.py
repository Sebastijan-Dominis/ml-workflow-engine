"""Integration tests for `pipelines.promotion.promote` CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pipelines.promotion.promote as promote_mod


def test_promote_main_invokes_promotion_service(tmp_path: Path, monkeypatch: Any) -> None:
    run_dir = tmp_path / "promotion_run"
    run_dir.mkdir()

    args = argparse.Namespace(
        problem="prob",
        segment="seg",
        version="v1",
        experiment_id="exp",
        train_run_id="train",
        eval_run_id="eval",
        explain_run_id="expl",
        stage="staging",
        logging_level="INFO",
    )

    monkeypatch.setattr(promote_mod, "parse_args", lambda: args)
    monkeypatch.setattr(promote_mod, "setup_logging", lambda *a, **k: None)

    def fake_build_context(args_obj):
        return SimpleNamespace(paths=SimpleNamespace(run_dir=run_dir))

    monkeypatch.setattr(promote_mod, "build_context", fake_build_context)

    called: dict[str, bool] = {"ran": False}

    class FakeService:
        def run(self, ctx):
            called["ran"] = True

    monkeypatch.setattr(promote_mod, "PromotionService", lambda: FakeService())

    rc = promote_mod.main()
    assert rc == 0
    assert called["ran"] is True
