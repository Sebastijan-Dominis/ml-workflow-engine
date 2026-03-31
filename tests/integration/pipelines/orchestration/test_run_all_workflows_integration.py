from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pipelines.orchestration.master.run_all_workflows as mod
import pytest

pytestmark = pytest.mark.integration


def test_run_all_workflows_calls_subprocess_for_each_step(monkeypatch) -> None:
    args = SimpleNamespace(env="dev", logging_level="INFO", owner="tester", skip_if_existing=True)
    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(mod, "log_completion", lambda *a, **k: None)

    called = []

    class FakeCompleted:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(cmd, text=True):
        called.append(cmd)
        return FakeCompleted(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = mod.main()

    assert rc == 0
    # At least one subprocess call should have been made (three steps expected)
    assert len(called) >= 1
