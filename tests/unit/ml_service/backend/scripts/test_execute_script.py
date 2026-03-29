import importlib
import types

from fastapi import HTTPException
from pydantic import BaseModel


def test_execute_script_builds_command_and_returns(monkeypatch):
    mod = importlib.import_module("ml_service.backend.scripts.execute_script")

    class Payload(BaseModel):
        foo: int | None = None
        flag: bool | None = None
        items: list[str] | None = None

    payload = Payload(foo=1, flag=True, items=["a", "b"])

    captured = {}

    def fake_run(cmd, capture_output, text, env, cwd):
        captured["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    # Replace the subprocess module used in the target module
    monkeypatch.setattr(mod, "subprocess", types.SimpleNamespace(run=fake_run))

    res = mod.execute_script("scripts.my_module", payload, boolean_args=["flag"])

    assert res["exit_code"] == 0
    assert res["stdout"] == "ok"

    # Ensure the assembled command contains expected pieces
    cmd = captured.get("cmd")
    assert cmd is not None
    assert cmd[0] == "python"
    assert "-m" in cmd
    assert "scripts.my_module" in cmd
    # flags
    assert "--foo" in cmd
    assert "1" in cmd
    assert "--flag" in cmd
    assert "True" in cmd
    # list expansion
    assert "--items" in cmd
    assert "a" in cmd and "b" in cmd


def test_execute_script_raises_http_on_subprocess_exception(monkeypatch):
    mod = importlib.import_module("ml_service.backend.scripts.execute_script")

    class Payload(BaseModel):
        x: int | None = None

    payload = Payload(x=1)

    def bad_run(*args, **kwargs):
        raise OSError("no exec")

    monkeypatch.setattr(mod, "subprocess", types.SimpleNamespace(run=bad_run))

    try:
        mod.execute_script("scripts.bad", payload)
        raised = False
    except HTTPException as e:
        raised = True
        assert e.status_code == 500

    assert raised
