"""Additional branch tests for `ml_service.frontend.docs.callbacks`."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_rewrite_links_outside_and_non_md(tmp_path: Path, monkeypatch) -> None:
    """Ensure non-markdown links and links resolving outside DOCS_ROOT are preserved."""
    mod = importlib.import_module("ml_service.frontend.docs.callbacks")

    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    monkeypatch.setattr(mod, "DOCS_ROOT", docs_root)

    # non-markdown link should be left unchanged
    md = "See [file](file.txt)"
    out = mod.rewrite_links(md, "readme.md")
    assert "file.txt" in out
    assert "/Docs?doc=" not in out

    # absolute path outside DOCS_ROOT should fall into the exception branch and be unchanged
    md2 = "See [X](/outside.md)"
    out2 = mod.rewrite_links(md2, "readme.md")
    assert "/outside.md" in out2
    assert "/Docs?doc=" not in out2
