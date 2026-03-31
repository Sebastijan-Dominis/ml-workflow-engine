import importlib

import pytest


def test_validate_config_payload_success(monkeypatch):
    mod = importlib.import_module(
        "ml_service.backend.configs.pipeline_cfg.validation.validate_config_payload"
    )

    class FakePipelineConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(mod, "PipelineConfig", FakePipelineConfig)

    payload = {"x": 1}
    validated = mod.validate_config_payload(payload)
    assert isinstance(validated, FakePipelineConfig)
    assert validated.kwargs == payload


def test_validate_config_payload_error_propagates(monkeypatch):
    mod = importlib.import_module(
        "ml_service.backend.configs.pipeline_cfg.validation.validate_config_payload"
    )

    def bad(**kwargs):
        raise ValueError("invalid config")

    monkeypatch.setattr(mod, "PipelineConfig", bad)

    with pytest.raises(ValueError) as exc:
        mod.validate_config_payload({"a": 2})

    assert "invalid config" in str(exc.value)
