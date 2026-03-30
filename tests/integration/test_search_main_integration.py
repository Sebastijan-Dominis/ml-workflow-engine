"""Integration tests for `pipelines.search.search.main`.

These tests exercise the high-level search CLI flow while stubbing heavy
components like the searcher and persistence to keep the test fast and
deterministic.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pipelines.search.search as search_mod
from ml.search.searchers.output import SearchOutput
from ml.types import AllSplitsInfo, SplitInfo


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        problem="p",
        segment="s",
        version="v",
        experiment_id=None,
        snapshot_binding_key=None,
        env="default",
        strict=True,
        logging_level="INFO",
        owner="me",
        clean_up_failure_management=False,
        overwrite_existing=False,
    )


def test_search_main_success(tmp_path: Path, monkeypatch: Any) -> None:
    """`main()` returns 0 and persists outputs when the searcher succeeds."""

    args = _make_args()
    monkeypatch.setattr(search_mod, "parse_args", lambda: args)

    # Run in an isolated cwd so experiment dirs land under tmp_path
    monkeypatch.chdir(tmp_path)

    # Simple config object and identity hashing
    fake_cfg = SimpleNamespace(algorithm=SimpleNamespace(value="catboost"))
    monkeypatch.setattr(search_mod, "load_and_validate_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(search_mod, "add_config_hash", lambda cfg: cfg)

    # Provide a fake searcher that returns a SearchOutput
    fake_output = SearchOutput(
        search_results={"x": 1},
        feature_lineage=[],
        pipeline_hash="ph",
        scoring_method="score",
        splits_info=AllSplitsInfo(
            train=SplitInfo(n_rows=0),
            val=SplitInfo(n_rows=0),
            test=SplitInfo(n_rows=0),
        ),
    )

    class FakeSearcher:
        def search(self, *a, **k):
            return fake_output

    monkeypatch.setattr(search_mod, "get_searcher", lambda key: FakeSearcher())

    persisted: dict[str, Any] = {}

    def fake_persist_experiment(*a, **k):
        persisted["called"] = True

    monkeypatch.setattr(search_mod, "persist_experiment", fake_persist_experiment)
    monkeypatch.setattr(search_mod, "delete_failure_management_folder", lambda *a, **k: None)
    monkeypatch.setattr(search_mod, "setup_logging", lambda *a, **k: None)

    rc = search_mod.main()
    assert rc == 0
    assert persisted.get("called", False) is True
