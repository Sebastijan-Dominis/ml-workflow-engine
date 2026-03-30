"""Integration test for the orchestration master `run_all_workflows` CLI."""

from typing import Any

import pipelines.orchestration.master.run_all_workflows as rw_mod


def test_run_all_workflows_main(monkeypatch: Any) -> None:
    called = {}

    def fake_main(*a, **k):
        called['ok'] = True
        return 0

    monkeypatch.setattr(rw_mod, 'main', fake_main)

    rc = rw_mod.main()
    assert rc == 0
    assert called['ok'] is True
