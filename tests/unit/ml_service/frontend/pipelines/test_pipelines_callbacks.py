import importlib
import types

import dash
import dash_bootstrap_components as dbc


def test_register_callbacks_toggle_and_run(monkeypatch):
    mod = importlib.import_module("ml_service.frontend.pipelines.callbacks")

    # Create a simple pipeline definition with boolean, number, text fields
    pipeline = {
        "name": "testpipe",
        "endpoint": "pipelines.test.run",
        "fields": [
            {"name": "flag", "type": "boolean"},
            {"name": "count", "type": "number"},
            {"name": "note", "type": "text"},
        ],
    }

    monkeypatch.setattr(mod, "FRONTEND_PIPELINES", [pipeline])

    # Fake app that records registered callbacks
    class FakeApp:
        def __init__(self):
            self._callbacks = []

        def callback(self, *args, **kwargs):
            def _decorator(func):
                self._callbacks.append(func)
                return func

            return _decorator

    fake_app = FakeApp()

    # Register callbacks (this will append two functions)
    mod.register_callbacks(fake_app)

    funcs = {f.__name__: f for f in fake_app._callbacks}
    assert "toggle_modal" in funcs
    assert "run_pipeline" in funcs

    toggle = funcs["toggle_modal"]
    run_pipeline = funcs["run_pipeline"]

    # Test toggle_modal: no trigger -> returns current state
    monkeypatch.setattr(dash, "callback_context", types.SimpleNamespace(triggered=[]))
    assert toggle(None, None, None, True) is True

    # simulate submit button triggered
    submit_id = f"/pipelines-{pipeline['name']}-submit"
    monkeypatch.setattr(dash, "callback_context", types.SimpleNamespace(triggered=[{"prop_id": f"{submit_id}.n_clicks"}]))
    assert toggle(1, None, None, False) is True

    # simulate confirm -> should close modal
    confirm_id = f"/pipelines-{pipeline['name']}-confirm"
    monkeypatch.setattr(dash, "callback_context", types.SimpleNamespace(triggered=[{"prop_id": f"{confirm_id}.n_clicks"}]))
    assert toggle(None, 1, None, True) is False

    # Test run_pipeline: if n_clicks is None -> no_update
    res = run_pipeline(None, True, "3", "hello")
    assert res is dash.no_update

    # Now test actual pipeline call paths with success and failure
    called = {}

    def fake_call(endpoint, payload):
        called["last_payload"] = payload
        return {"status": "SUCCESS"}

    monkeypatch.setattr(mod, "call_pipeline", fake_call)

    out = run_pipeline(1, True, "2", "text")
    # should return a Textarea component with success background
    assert isinstance(out, dbc.Textarea)
    assert getattr(out, "id", None) == f"/pipelines-{pipeline['name']}-result"
    style = getattr(out, "style", {}) or {}
    assert style.get("backgroundColor") == "#81ff81"

    # failing pipeline
    def fake_call_fail(endpoint, payload):
        called["last_payload"] = payload
        return {"status": "FAIL"}

    monkeypatch.setattr(mod, "call_pipeline", fake_call_fail)
    out2 = run_pipeline(1, False, "3.5", "")
    assert isinstance(out2, dbc.Textarea)
    style2 = getattr(out2, "style", {}) or {}
    assert style2.get("backgroundColor") == "#ff8181"
