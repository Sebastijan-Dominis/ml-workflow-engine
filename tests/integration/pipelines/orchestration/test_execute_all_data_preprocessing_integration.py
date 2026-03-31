from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pipelines.orchestration.data.execute_all_data_preprocessing as mod
import pytest

pytestmark = pytest.mark.integration


def test_execute_all_data_preprocessing_runs_expected_subprocesses(tmp_path: Path, monkeypatch: Any) -> None:
    # Create a minimal repo-like structure under tmp_path
    data_raw_snap = tmp_path / "data" / "raw" / "mydata" / "v1" / "snap1"
    data_raw_snap.mkdir(parents=True)

    interim_cfg_dir = tmp_path / "configs" / "data" / "interim" / "mydata"
    interim_cfg_dir.mkdir(parents=True)
    (interim_cfg_dir / "v1.yaml").write_text("dummy: true")

    processed_cfg_dir = tmp_path / "configs" / "data" / "processed" / "mydata"
    processed_cfg_dir.mkdir(parents=True)
    (processed_cfg_dir / "v1.yaml").write_text("dummy: true")

    # Force the orchestrator to run by disabling skip-if-existing
    args = SimpleNamespace(skip_if_existing=False)
    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(mod, "log_completion", lambda *a, **k: None)

    # Run from tmp_path so relative paths in the module resolve to our test tree
    monkeypatch.chdir(tmp_path)

    called: list[list[str]] = []

    class FakeResult:
        def __init__(self, stdout: str = "ok") -> None:
            self.stdout = stdout
            self.returncode = 0

    def fake_run(cmd, check=True, capture_output=True, text=True):
        called.append(list(cmd))
        return FakeResult(stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = mod.main()

    assert rc == 0
    # Expect at least three subprocess calls: register_raw_snapshot, build_interim_dataset, build_processed_dataset
    assert len(called) >= 3
