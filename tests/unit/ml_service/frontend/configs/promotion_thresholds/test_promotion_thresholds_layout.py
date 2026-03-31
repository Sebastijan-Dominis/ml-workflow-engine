import importlib


def _collect_ids(obj: object) -> set:
    ids: set = set()
    if hasattr(obj, "id"):
        maybe_id = getattr(obj, "id", None)
        if maybe_id:
            ids.add(maybe_id)
    props = getattr(obj, "props", None)
    if isinstance(props, dict):
        pid = props.get("id")
        if pid:
            ids.add(pid)

    children = getattr(obj, "children", None)
    if children is None:
        return ids
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is None:
            continue
        ids.update(_collect_ids(child))
    return ids


def test_promotion_thresholds_layout_and_page_ids():
    mod = importlib.import_module(
        "ml_service.frontend.configs.promotion_thresholds.layout"
    )
    layout = mod.build_layout()
    ids = _collect_ids(layout)
    assert f"{mod.PAGE_PREFIX}-config-editor" in ids
    assert f"{mod.PAGE_PREFIX}-validate-btn" in ids
    assert f"{mod.PAGE_PREFIX}-confirm-modal" in ids
    assert f"{mod.PAGE_PREFIX}-problem-type-input" in ids
    assert f"{mod.PAGE_PREFIX}-segment-input" in ids


def test_promotion_thresholds_page_registers(monkeypatch):
    mod = importlib.import_module(
        "ml_service.frontend.configs.promotion_thresholds.page"
    )
    called = []

    def fake_register(app):
        called.append(True)

    # patch internal register_callbacks to avoid heavy dash objects
    monkeypatch.setattr(
        "ml_service.frontend.configs.promotion_thresholds.page.register_callbacks",
        fake_register,
        raising=False,
    )

    # calling register should call our fake
    mod.register(object())
    assert called
