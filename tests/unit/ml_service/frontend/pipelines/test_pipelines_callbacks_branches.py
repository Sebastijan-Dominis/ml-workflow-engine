"""Additional pipeline callback branches to reach full coverage."""

from __future__ import annotations

import importlib
import types


def _make_pipeline():
    return {
        "name": "testpipe",
        "endpoint": "pipelines.test.run",
        "fields": [
            {"name": "flag", "type": "boolean"},
            {"name": "count", "type": "number"},
            {"name": "note", "type": "text"},
        ],
    }


def test_toggle_modal_unmatched_button(monkeypatch):
    mod = importlib.import_module("ml_service.frontend.pipelines.callbacks")
    pipeline = _make_pipeline()
    monkeypatch.setattr(mod, "FRONTEND_PIPELINES", [pipeline])

    class FakeApp:
        def __init__(self):
            self._callbacks = []

        def callback(self, *args, **kwargs):
            def _decorator(f):
                self._callbacks.append(f)
                return f

            return _decorator

    fake_app = FakeApp()
    mod.register_callbacks(fake_app)
    funcs = {f.__name__: f for f in fake_app._callbacks}
    toggle = funcs["toggle_modal"]

    # simulate an unrelated trigger -> should return current is_open
    monkeypatch.setattr(mod.dash, "callback_context", types.SimpleNamespace(triggered=[{"prop_id": "other.id.n_clicks"}]))
    assert toggle(None, None, None, True) is True


def test_run_pipeline_boolean_none_and_number_empty(monkeypatch):
    mod = importlib.import_module("ml_service.frontend.pipelines.callbacks")
    pipeline = _make_pipeline()
    monkeypatch.setattr(mod, "FRONTEND_PIPELINES", [pipeline])

    class FakeApp:
        def __init__(self):
            self._callbacks = []

        def callback(self, *args, **kwargs):
            def _decorator(f):
                self._callbacks.append(f)
                return f

            return _decorator

    fake_app = FakeApp()
    mod.register_callbacks(fake_app)
    funcs = {f.__name__: f for f in fake_app._callbacks}
    run_pipeline = funcs["run_pipeline"]

    called = {}

    def fake_call(endpoint, payload):
        called["payload"] = payload
        return {"status": "SUCCESS"}

    monkeypatch.setattr(mod, "call_pipeline", fake_call)
    # trigger execution (confirm button clicked)
    monkeypatch.setattr(mod.dash, "callback_context", types.SimpleNamespace(triggered=[{"prop_id": f"/pipelines-{pipeline['name']}-confirm.n_clicks"}]))
    out = run_pipeline(1, None, "", "text")

    # boolean None -> False, number '' -> None, text -> 'text'
    assert called["payload"] == {"flag": False, "count": None, "note": "text"}

    import dash_bootstrap_components as dbc

    assert isinstance(out, dbc.Textarea)
