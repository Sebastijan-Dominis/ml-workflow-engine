"""Tests for `ml_service.frontend.scripts.callbacks` behavior."""

from __future__ import annotations

import importlib
import types
from types import SimpleNamespace
from typing import Any

import dash
import dash_bootstrap_components as dbc


class DummyApp:
    def __init__(self) -> None:
        self.callbacks: list[dict[str, Any]] = []

    def callback(self, *a: Any, **k: Any):
        def dec(f: Any) -> Any:
            self.callbacks.append({"func": f, "args": a, "kwargs": k})
            return f

        return dec


def _find(app: DummyApp, name: str):
    for c in app.callbacks:
        if c["func"].__name__ == name:
            return c["func"]
    raise AssertionError(name)


def test_toggle_modal_and_run_pipeline(monkeypatch) -> None:
    cb_mod = importlib.import_module("ml_service.frontend.scripts.callbacks")

    custom = [
        {"name": "x", "endpoint": "/scripts/x", "fields": [
            {"name": "n", "type": "number"},
            {"name": "flag", "type": "boolean"},
            {"name": "ops", "type": "text"},
        ]}
    ]

    # patch the FRONTEND_SCRIPTS used by callbacks
    monkeypatch.setattr(cb_mod, "FRONTEND_SCRIPTS", custom)

    app = DummyApp()
    cb_mod.register_callbacks(app)

    toggle = _find(app, "toggle_modal")
    run = _find(app, "run_pipeline")

    # simulate dash.callback_context.triggered
    fake_ctx = SimpleNamespace(triggered=[{"prop_id": "/scripts-x-submit.n_clicks"}])
    monkeypatch.setattr(cb_mod.dash, "callback_context", fake_ctx, raising=False)

    # submit click should open modal
    assert toggle(1, None, None, False) is True

    # confirm click should close modal
    fake_ctx.triggered = [{"prop_id": "/scripts-x-confirm.n_clicks"}]
    assert toggle(None, 1, None, True) is False

    # run pipeline: monkeypatch call_script to return SUCCESS
    monkeypatch.setattr(cb_mod, "call_script", lambda ep, payload: {"status": "SUCCESS"})
    res = run(1, "3", True, "a,b")
    # returned is a dbc.Textarea-like object; its `value` should contain the status
    assert "SUCCESS" in str(res)


def test_register_scripts_toggle_and_run(monkeypatch):
    mod = importlib.import_module("ml_service.frontend.scripts.callbacks")

    script = {
        "name": "testscript",
        "endpoint": "scripts.test.endpoint",
        "fields": [
            {"name": "operators", "type": "text"},
            {"name": "flag", "type": "boolean"},
            {"name": "count", "type": "number"},
            {"name": "note", "type": "text"},
        ],
    }

    monkeypatch.setattr(mod, "FRONTEND_SCRIPTS", [script])

    class FakeApp:
        def __init__(self):
            self._callbacks = []

        def callback(self, *args, **kwargs):
            def _decorator(func):
                self._callbacks.append(func)
                return func

            return _decorator

    fake_app = FakeApp()
    mod.register_callbacks(fake_app)

    funcs = {f.__name__: f for f in fake_app._callbacks}
    assert "toggle_modal" in funcs
    assert "run_pipeline" in funcs

    toggle = funcs["toggle_modal"]
    run_pipeline = funcs["run_pipeline"]

    # no trigger -> return current
    monkeypatch.setattr(dash, "callback_context", types.SimpleNamespace(triggered=[]))
    assert toggle(None, None, None, True) is True

    submit_id = f"{mod.PAGE_PREFIX}-{script['name']}-submit"
    monkeypatch.setattr(dash, "callback_context", types.SimpleNamespace(triggered=[{"prop_id": f"{submit_id}.n_clicks"}]))
    assert toggle(1, None, None, False) is True

    confirm_id = f"{mod.PAGE_PREFIX}-{script['name']}-confirm"
    monkeypatch.setattr(dash, "callback_context", types.SimpleNamespace(triggered=[{"prop_id": f"{confirm_id}.n_clicks"}]))
    assert toggle(None, 1, None, True) is False

    # n_clicks None -> no update
    res = run_pipeline(None, "a,b", True, "3", "hello")
    assert res is dash.no_update

    captured = {}

    def fake_call(endpoint, payload):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"status": "SUCCESS"}

    monkeypatch.setattr(mod, "call_script", fake_call)

    out = run_pipeline(1, "a,b", True, "3", "hello")
    assert isinstance(out, dbc.Textarea)
    assert getattr(out, "id", None) == f"{mod.PAGE_PREFIX}-{script['name']}-result"
    style = getattr(out, "style", {}) or {}
    assert style.get("backgroundColor") == "#81ff81"

    # failing status
    def fake_call_fail(endpoint, payload):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"status": "FAIL"}

    monkeypatch.setattr(mod, "call_script", fake_call_fail)
    out2 = run_pipeline(1, "a,b", True, "3", "")
    assert isinstance(out2, dbc.Textarea)
    style2 = getattr(out2, "style", {}) or {}
    assert style2.get("backgroundColor") == "#ff8181"
