import importlib


class DummyApp:
    def __init__(self):
        self.callbacks = []

    def callback(self, *args, **kwargs):
        def dec(func):
            self.callbacks.append(func)
            return func

        return dec


class FakeResp:
    def __init__(self, ok=True, json_data=None, status=200, text=""):
        self.ok = ok
        self._json = json_data or {}
        self.status_code = status
        self.text = text

    def json(self):
        return self._json


def test_pipeline_cfg_validate_and_write(monkeypatch):
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")

    app = DummyApp()
    mod.register_callbacks(app)
    # should have registered two callbacks
    names = {fn.__name__ for fn in app.callbacks}
    assert "validate_config" in names
    assert "write_config" in names

    # find functions
    validate_fn = next(fn for fn in app.callbacks if fn.__name__ == "validate_config")
    write_fn = next(fn for fn in app.callbacks if fn.__name__ == "write_config")

    # missing inputs -> error
    res = validate_fn(None, None, "alg", "")
    assert res[1] is False

    # backend success path for validate
    monkeypatch.setattr(
        "ml_service.frontend.configs.pipeline_cfg.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=True, json_data={"valid": True, "exists": False, "normalized": {}}),
    )
    res2 = validate_fn(None, "dt", "alg", "version: 1")
    assert res2[1] is True

    # write: backend reports exists
    monkeypatch.setattr(
        "ml_service.frontend.configs.pipeline_cfg.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=True, json_data={"status": "exists", "message": "exists"}),
    )
    res3 = write_fn(None, "dt", "alg", "version: 1")
    assert res3[1] is False
