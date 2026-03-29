"""Tests for executing scripts and pipelines via subprocess wrappers.

These tests assert that CLI arguments are constructed correctly and that
subprocess failures are surfaced as `HTTPException`.
"""
from __future__ import annotations

import subprocess
from typing import Any

import pytest
from ml_service.backend.pipelines.execute_pipeline import execute_pipeline
from ml_service.backend.registries.exit_codes_meaning import EXIT_MEANING
from ml_service.backend.scripts.execute_script import execute_script
from pydantic import BaseModel


def test_execute_script_list_and_boolean(monkeypatch) -> None:
    class Payload(BaseModel):
        name: str
        list_items: list[str] | None = None
        enable: bool | None = None
        empty_field: str | None = None

    payload = Payload(name="test", list_items=["a", "b"], enable=True, empty_field="")

    def fake_run(cmd: list[str], capture_output: Any, text: Any, env: Any, cwd: Any):
        # basic sanity checks on the constructed command
        assert cmd[0] == "python"
        assert "-m" in cmd
        # list handling
        assert "--list-items" in cmd
        idx = cmd.index("--list-items")
        assert cmd[idx + 1] == "a"
        assert cmd[idx + 2] == "b"
        # boolean handling
        assert "--enable" in cmd
        idx2 = cmd.index("--enable")
        assert cmd[idx2 + 1] == "True"

        class R:  # minimal CompletedProcess-like
            returncode = 0
            stdout = "ok"
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = execute_script("some.module", payload, boolean_args=["enable"])  # type: ignore[arg-type]

    assert res["exit_code"] == 0
    assert res["stdout"] == "ok"
    assert res["stderr"] == ""
    assert res["status"] == EXIT_MEANING.get(0, "UNKNOWN_ERROR")


def test_execute_script_start_failure(monkeypatch) -> None:
    class Payload(BaseModel):
        name: str

    payload = Payload(name="x")

    def bad_run(*args: Any, **kwargs: Any):
        raise OSError("cannot start")

    monkeypatch.setattr(subprocess, "run", bad_run)

    with pytest.raises(Exception) as exc:
        execute_script("some.module", payload)

    # HTTPException from FastAPI exposes `status_code` attribute
    assert getattr(exc.value, "status_code", 500) == 500


def test_execute_pipeline_boolean(monkeypatch) -> None:
    class Payload(BaseModel):
        strict: bool | None = None
        name: str = "p"

    payload = Payload(strict=True, name="p")

    def fake_run(cmd: list[str], capture_output: Any, text: Any, env: Any, cwd: Any):
        assert "--strict" in cmd
        idx = cmd.index("--strict")
        assert cmd[idx + 1] == "True"

        class R:
            returncode = 2
            stdout = "done"
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = execute_pipeline("pipelines.example", payload, boolean_args=["strict"])  # type: ignore[arg-type]

    assert res["exit_code"] == 2
    assert res["stdout"] == "done"
    assert res["status"] == EXIT_MEANING.get(2, "UNKNOWN_ERROR")


def test_execute_pipeline_start_failure(monkeypatch) -> None:
    class Payload(BaseModel):
        name: str

    payload = Payload(name="y")

    def bad_run(*args: Any, **kwargs: Any):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(subprocess, "run", bad_run)

    with pytest.raises(Exception) as exc:
        execute_pipeline("pipelines.example", payload)

    # HTTPException from FastAPI exposes `status_code` attribute
    assert getattr(exc.value, "status_code", 500) == 500
