import importlib
from pathlib import Path


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
