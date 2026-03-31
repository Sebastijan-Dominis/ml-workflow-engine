import importlib
from types import SimpleNamespace


def test_dir_viewer_load_branches(monkeypatch):
    mod = importlib.import_module("ml_service.frontend.dir_viewer.callbacks")

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
    assert "load_dir" in funcs
    load_dir = funcs["load_dir"]

    # Missing path -> error alert
    out = load_dir(None, "")
    assert out[0] == ""
    assert out[1] == "yaml"
    alert = out[2]
    children = getattr(alert, "children", "")
    assert "Path required" in str(children)

    # Backend unreachable
    def raise_post(*a, **k):
        raise Exception("conn")

    monkeypatch.setattr(mod, "requests", SimpleNamespace(post=raise_post))
    out2 = load_dir(None, "sub")
    assert "Backend unreachable" in str(getattr(out2[2], "children", ""))

    # Not OK response
    class Resp:
        ok = False
        status_code = 403
        text = "Not allowed"

    monkeypatch.setattr(mod, "requests", SimpleNamespace(post=lambda *a, **k: Resp()))
    out3 = load_dir(None, "sub")
    assert "403" in str(getattr(out3[2], "children", ""))

    # Success
    class RespOK:
        ok = True

        def json(self):
            return {"tree_yaml": "a: b\n", "path": "/tmp/sub"}

    monkeypatch.setattr(mod, "requests", SimpleNamespace(post=lambda *a, **k: RespOK()))
    out4 = load_dir(None, "sub")
    assert out4[0] == "a: b\n"
    assert out4[1] == "yaml"
    assert "Loaded directory tree for /tmp/sub" in str(getattr(out4[2], "children", ""))
