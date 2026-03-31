"""Unit tests for ``ml_service.backend.pipelines.execute_pipeline``.

These tests cover subprocess start failures and non-zero exit-code paths,
including that boolean flags are passed as string values when requested.
"""

from __future__ import annotations

import importlib
import types
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from ml_service.backend.pipelines.execute_pipeline import execute_pipeline
from ml_service.backend.registries.exit_codes_meaning import EXIT_MEANING
from pydantic import BaseModel


class DummyPayload(BaseModel):
    text: str | None = None
    flag: bool | None = None


def test_execute_pipeline_subprocess_raises(monkeypatch: Any) -> None:
    """If starting the subprocess raises, return an HTTP 500 error."""

    def fake_run(*_a: Any, **_k: Any) -> None:  # pragma: no cover - exercised
        raise Exception("spawn failed")

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(HTTPException) as excinfo:
        execute_pipeline("ml_service.pipelines.fake", DummyPayload(text="a"))
    assert excinfo.value.status_code == 500


def test_execute_pipeline_nonzero_and_flag_in_cmd(monkeypatch: Any) -> None:
    """When subprocess returns non-zero, the mapping is used and the command
    contains boolean flags converted to strings when requested.
    """

    last_cmd: dict[str, Any] = {}

    def fake_run(cmd: list[str], capture_output: bool, text: bool, env: dict[str, str], cwd: str):
        last_cmd["cmd"] = cmd
        return SimpleNamespace(returncode=2, stdout="out", stderr="err")

    monkeypatch.setattr("subprocess.run", fake_run)
    payload = DummyPayload(text="hello", flag=True)
    res = execute_pipeline("ml_service.pipelines.fake", payload, boolean_args=["flag"])
    assert res["exit_code"] == 2
    assert res["status"] == EXIT_MEANING.get(2, "UNKNOWN_ERROR")
    # boolean flag was added as a CLI flag (e.g. --flag True)
    assert any(part == "--flag" or part.startswith("--flag") for part in (str(p) for p in last_cmd["cmd"]))


def test_execute_pipeline_builds_command_and_returns(monkeypatch):
    mod = importlib.import_module("ml_service.backend.pipelines.execute_pipeline")

    class Payload(BaseModel):
        foo: int | None = None
        flag: bool | None = None

    payload = Payload(foo=1, flag=True)

    captured = {}

    def fake_run(cmd, capture_output, text, env, cwd):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "subprocess", types.SimpleNamespace(run=fake_run))

    res = mod.execute_pipeline("pipelines.my_module", payload, boolean_args=["flag"])

    assert res["exit_code"] == 0
    assert res["stdout"] == "ok"

    cmd = captured.get("cmd")
    assert cmd is not None
    assert cmd[0] == "python"
    assert "-m" in cmd
    assert "pipelines.my_module" in cmd
    assert "--foo" in cmd
    assert "1" in cmd
    assert "--flag" in cmd
    assert "True" in cmd


def test_execute_pipeline_raises_http_on_subprocess_exception(monkeypatch):
    mod = importlib.import_module("ml_service.backend.pipelines.execute_pipeline")

    class Payload(BaseModel):
        x: int | None = None

    payload = Payload(x=1)

    def bad_run(*args, **kwargs):
        raise OSError("no exec")

    monkeypatch.setattr(mod, "subprocess", types.SimpleNamespace(run=bad_run))

    try:
        mod.execute_pipeline("pipelines.bad", payload)
        raised = False
    except HTTPException as e:
        raised = True
        assert e.status_code == 500

    assert raised


def test_execute_pipeline_skips_empty_and_none(monkeypatch):
    mod = importlib.import_module("ml_service.backend.pipelines.execute_pipeline")

    class Payload(BaseModel):
        foo: int | None = None
        bar: str | None = None
        baz: int | None = None

    payload = Payload(foo=None, bar="", baz=3)

    captured = {}

    def fake_run(cmd, capture_output, text, env, cwd):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "subprocess", types.SimpleNamespace(run=fake_run))

    res = mod.execute_pipeline("pipelines.my_module", payload)
    assert res["exit_code"] == 0
    cmd = captured.get("cmd")
    assert cmd is not None
    assert "--baz" in cmd
    assert not any(part == "--foo" or part.startswith("--foo") for part in cmd)
    assert not any(part == "--bar" or part.startswith("--bar") for part in cmd)


def test_execute_pipeline_boolean_false_included(monkeypatch):
    mod = importlib.import_module("ml_service.backend.pipelines.execute_pipeline")

    class Payload(BaseModel):
        flag: bool | None = None

    payload = Payload(flag=False)

    captured = {}

    def fake_run(cmd, capture_output, text, env, cwd):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "subprocess", types.SimpleNamespace(run=fake_run))

    res = mod.execute_pipeline("pipelines.my_module", payload, boolean_args=["flag"])
    assert res["exit_code"] == 0
    cmd = captured.get("cmd")
    assert cmd is not None
    assert "--flag" in cmd
    assert "False" in cmd
