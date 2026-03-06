"""Unit tests for promotion CLI entrypoint behavior."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipelines.promotion import promote as module

pytestmark = pytest.mark.unit


def test_parse_args_parses_required_and_optional_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse required promotion identifiers and default optional logging level."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "promote.py",
            "--problem",
            "no_show",
            "--segment",
            "global",
            "--version",
            "v1",
            "--experiment-id",
            "exp-1",
            "--train-run-id",
            "train-1",
            "--eval-run-id",
            "eval-1",
            "--explain-run-id",
            "explain-1",
            "--stage",
            "staging",
        ],
    )

    args = module.parse_args()

    assert args.problem == "no_show"
    assert args.segment == "global"
    assert args.version == "v1"
    assert args.experiment_id == "exp-1"
    assert args.train_run_id == "train-1"
    assert args.eval_run_id == "eval-1"
    assert args.explain_run_id == "explain-1"
    assert args.stage == "staging"
    assert args.logging_level == "INFO"


def test_main_returns_zero_and_runs_service_when_promotion_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build context, initialize logging, and return success exit code on normal run."""
    args = argparse.Namespace(
        problem="no_show",
        segment="global",
        version="v1",
        experiment_id="exp-1",
        train_run_id="train-1",
        eval_run_id="eval-1",
        explain_run_id="explain-1",
        stage="production",
        logging_level="debug",
    )
    context = SimpleNamespace(paths=SimpleNamespace(run_dir=Path("runs") / "promotion" / "run-1"))

    captured: dict[str, Any] = {"ran": False}

    class _Service:
        def run(self, run_context: Any) -> None:
            captured["ran"] = True
            captured["run_context"] = run_context

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "build_context", lambda parsed_args: context if parsed_args is args else None)
    monkeypatch.setattr(module, "PromotionService", _Service)

    def _setup_logging(log_file: Path, log_level: int) -> None:
        captured["log_file"] = log_file
        captured["log_level"] = log_level

    monkeypatch.setattr(module, "setup_logging", _setup_logging)

    result = module.main()

    assert result == 0
    assert captured["ran"] is True
    assert captured["run_context"] is context
    assert captured["log_file"] == context.paths.run_dir / "promotion.log"
    assert captured["log_level"] == logging.DEBUG


def test_main_maps_errors_to_exit_codes_via_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return resolver-provided exit code when promotion service raises an exception."""
    args = argparse.Namespace(
        problem="no_show",
        segment="global",
        version="v1",
        experiment_id="exp-1",
        train_run_id="train-1",
        eval_run_id="eval-1",
        explain_run_id="explain-1",
        stage="production",
        logging_level="INFO",
    )
    context = SimpleNamespace(paths=SimpleNamespace(run_dir=Path("runs") / "promotion" / "run-err"))

    class _ExpectedError(RuntimeError):
        pass

    class _FailingService:
        def run(self, run_context: Any) -> None:
            _ = run_context
            raise _ExpectedError("boom")

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "build_context", lambda _args: context)
    monkeypatch.setattr(module, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "PromotionService", _FailingService)

    seen: dict[str, Any] = {}

    def _resolve_exit_code(error: Exception) -> int:
        seen["error"] = error
        return 27

    monkeypatch.setattr(module, "resolve_exit_code", _resolve_exit_code)

    result = module.main()

    assert result == 27
    assert isinstance(seen["error"], _ExpectedError)
