"""Cover the branch in `generate_page_links` where a page has no icon."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import ml_service.frontend.app as app_mod


def test_generate_page_links_no_icon(monkeypatch):
    # Create a modified PAGES mapping that includes a page not present in ICONS
    original = app_mod.PAGES
    try:
        new_pages = dict(original)
        new_pages["No Icon Page"] = lambda: dbc.Container("noop")
        monkeypatch.setattr(app_mod, "PAGES", new_pages)

        links = app_mod.generate_page_links()

        found = False
        for link in links:
            lid = getattr(link, "id", None)
            if lid == "nav-No_Icon_Page":
                found = True
                break
            props = getattr(link, "props", None)
            if isinstance(props, dict) and props.get("id") == "nav-No_Icon_Page":
                found = True
                break

        assert found, "Expected nav-No_Icon_Page in generated links"
    finally:
        monkeypatch.setattr(app_mod, "PAGES", original)
