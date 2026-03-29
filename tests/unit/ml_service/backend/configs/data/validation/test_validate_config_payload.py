import importlib

import pytest


def test_validate_config_payload_interim(monkeypatch):
    vmod = importlib.import_module(
        "ml_service.backend.configs.data.validation.validate_config_payload"
    )

    class FakeInterim:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(vmod, "InterimConfig", FakeInterim)
    res = vmod.validate_config_payload("interim", {"a": 1})
    assert isinstance(res, FakeInterim)
    assert res.kwargs == {"a": 1}


def test_validate_config_payload_processed(monkeypatch):
    vmod = importlib.import_module(
        "ml_service.backend.configs.data.validation.validate_config_payload"
    )

    class FakeProcessed:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(vmod, "ProcessedConfig", FakeProcessed)
    res = vmod.validate_config_payload("processed", {"b": 2})
    assert isinstance(res, FakeProcessed)
    assert res.kwargs == {"b": 2}


def test_validate_config_payload_unknown():
    vmod = importlib.import_module(
        "ml_service.backend.configs.data.validation.validate_config_payload"
    )
    with pytest.raises(ValueError):
        vmod.validate_config_payload("unknown", {})
