"""Tests for `ml_service.frontend.docs.callbacks` functions."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_rewrite_links_and_load(tmp_path: Path, monkeypatch) -> None:
    mod = importlib.import_module("ml_service.frontend.docs.callbacks")

    # prepare docs tree
    docs_root = tmp_path
    a = docs_root / "a.md"
    b = docs_root / "b.md"
    a.write_text("See [B](b.md)")
    b.write_text("Hello B")

    monkeypatch.setattr(mod, "DOCS_ROOT", docs_root)

    out = mod.rewrite_links("See [B](b.md)", "a.md")
    assert "/Docs?doc=b.md" in out

    # test load_doc_from_url when document exists by registering callbacks
    class DummyApp:
        def __init__(self):
            self._callbacks = {}

        def callback(self, *args, **kwargs):
            def decorator(f):
                self._callbacks[f.__name__] = f
                return f

            return decorator

    app = DummyApp()
    mod.register_callbacks(app)
    load_fn = app._callbacks.get("load_doc_from_url")
    assert load_fn is not None
    res = load_fn("?doc=a.md")
    assert "Docs?doc=b.md" in res or "See" in res


def test_load_doc_not_found(monkeypatch, tmp_path: Path) -> None:
    mod = importlib.import_module("ml_service.frontend.docs.callbacks")
    monkeypatch.setattr(mod, "DOCS_ROOT", tmp_path)
    class DummyApp:
        def __init__(self):
            self._callbacks = {}

        def callback(self, *args, **kwargs):
            def decorator(f):
                self._callbacks[f.__name__] = f
                return f

            return decorator

    app = DummyApp()
    mod.register_callbacks(app)
    load_fn = app._callbacks.get("load_doc_from_url")
    assert load_fn is not None
    assert load_fn("?doc=nope.md") == "Document not found."


def test_rewrite_links_internal_and_external(tmp_path, monkeypatch):
    mod = importlib.import_module("ml_service.frontend.docs.callbacks")

    # set docs root to tmp_path and create files
    monkeypatch.setattr(mod, "DOCS_ROOT", Path(tmp_path))

    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "other.md").write_text("# other")

    md = "See [other](other.md) and [ext](http://example.com)"
    out = mod.rewrite_links(md, "subdir/readme.md")
    assert "(/Docs?doc=subdir/other.md)" in out
    assert "http://example.com" in out
