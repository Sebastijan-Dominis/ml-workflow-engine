"""Tests for the main Dash app helpers in ml_service.frontend.app."""

from __future__ import annotations

import ml_service.frontend.app as app_mod


def _has_children(obj: object) -> bool:
    return hasattr(obj, "children") or hasattr(obj, "props")


def test_generate_page_links():
    links = app_mod.generate_page_links()
    assert isinstance(links, list)
    assert len(links) == len(app_mod.PAGES)
    assert all(_has_children(link) for link in links)


def test_toggle_and_close_sidebar():
    toggle = getattr(app_mod.toggle_sidebar, "__wrapped__", app_mod.toggle_sidebar)
    assert toggle(1, True) is False
    assert toggle(2, False) is True
    assert toggle(0, True) is True

    close = getattr(app_mod.close_sidebar_on_home, "__wrapped__", app_mod.close_sidebar_on_home)
    assert close(1, True) is False
    assert close(None, True) is True


def test_display_page_and_404():
    display = getattr(app_mod.display_page, "__wrapped__", app_mod.display_page)
    # root -> home
    res_home = display("/")
    assert _has_children(res_home)
    # known page
    res_page = display("/Pipelines")
    assert _has_children(res_page)
    # unknown page -> 404 container with H2 child
    res_404 = display("/this_page_does_not_exist")
    txt = ""
    if hasattr(res_404, "children"):
        ch = res_404.children
        if isinstance(ch, (list, tuple)):
            ch = ch[0]
        if hasattr(ch, "children"):
            if isinstance(ch.children, (list, tuple)):
                txt = "".join(str(x) for x in ch.children)
            else:
                txt = str(ch.children)
    assert "404" in txt


def test_update_active_links():
    update = getattr(app_mod.update_active_links, "__wrapped__", app_mod.update_active_links)
    res_none = update(None)
    assert all(v is False for v in res_none)
    res_pipelines = update("/Pipelines")
    expected = [name == "Pipelines" for name in app_mod.PAGES]
    assert res_pipelines == expected
