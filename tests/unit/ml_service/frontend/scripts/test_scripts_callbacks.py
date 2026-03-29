import importlib
import types

import dash
import dash_bootstrap_components as dbc


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
