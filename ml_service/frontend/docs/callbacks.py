"""Callbacks for the documentation viewer page."""

import re
from urllib.parse import parse_qs

from dash import Input, Output

from .layout import DOCS_ROOT


def rewrite_links(md_text: str, current_doc: str) -> str:
    """
    Convert relative markdown links to Dash routes, resolving paths properly.
    """
    current_path = (DOCS_ROOT / current_doc).resolve()
    current_dir = current_path.parent

    def replacer(match):
        text, link = match.groups()

        # Skip external links
        if link.startswith("http") or link.startswith("#"):
            return match.group(0)

        # Only handle markdown files
        if link.endswith(".md"):
            try:
                resolved = (current_dir / link).resolve()
                relative = resolved.relative_to(DOCS_ROOT.resolve())
                return f"[{text}](/Docs?doc={relative.as_posix()})"
            except Exception:
                return match.group(0)

        return match.group(0)

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replacer, md_text)


def register_callbacks(app):
    """Registers callbacks for the docs page."""

    @app.callback(
        Output("doc-content", "children"),
        Input("url", "search"),
    )
    def load_doc_from_url(search):
        if not search:
            doc_path = "readme.md"
        else:
            query = parse_qs(search.lstrip("?"))
            doc_path = query.get("doc", ["readme.md"])[0]

        full_path = DOCS_ROOT / doc_path

        if not full_path.exists():
            return "Document not found."

        content = full_path.read_text(encoding="utf-8")
        return rewrite_links(content, doc_path)
