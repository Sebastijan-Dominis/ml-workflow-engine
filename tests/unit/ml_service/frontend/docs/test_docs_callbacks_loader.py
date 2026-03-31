import importlib


def test_rewrite_links_and_load_doc(tmp_path, monkeypatch):
    mod = importlib.import_module("ml_service.frontend.docs.callbacks")

    # Patch DOCS_ROOT to tmp_path
    monkeypatch.setattr(mod, "DOCS_ROOT", tmp_path)

    # Create docs structure
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "other.md").write_text("# Other\nContent")

    readme = tmp_path / "readme.md"
    readme.write_text("See [other](sub/other.md) and [web](http://example.com) and [anch](#sec)")

    # Test rewrite_links directly
    out = mod.rewrite_links(readme.read_text(), "readme.md")
    assert "/Docs?doc=sub/other.md" in out
    assert "http://example.com" in out
    assert "#sec" in out

    # Register callbacks and extract loader
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
    assert "load_doc_from_url" in funcs

    loader = funcs["load_doc_from_url"]

    # No search -> readme.md
    res = loader("")
    assert "/Docs?doc=sub/other.md" in res

    # Non-existing doc -> not found
    res2 = loader("?doc=missing.md")
    assert "Document not found" in res2
