import importlib

import pytest


def test_validate_all_configs_success(monkeypatch):
    v = importlib.import_module(
        "ml_service.backend.configs.modeling.validation.validate_all_configs"
    )

    class FakeModelSpecs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


    class FakeSearch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


    class FakeTrain:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


    monkeypatch.setattr(v, "ModelSpecs", FakeModelSpecs)
    monkeypatch.setattr(v, "SearchConfigForValidation", FakeSearch)
    monkeypatch.setattr(v, "TrainConfigForValidation", FakeTrain)

    from ml_service.backend.configs.modeling.models.configs import RawConfigsWithLineage

    raw = RawConfigsWithLineage(model_specs={"a": 1}, search={"b": 2}, training={"c": 3})
    validated = v.validate_all_configs(raw)

    assert validated.model_specs.kwargs == {"a": 1}
    assert validated.search.kwargs == {"b": 2}
    assert validated.training.kwargs == {"c": 3}


def test_validate_all_configs_error(monkeypatch):
    v = importlib.import_module(
        "ml_service.backend.configs.modeling.validation.validate_all_configs"
    )

    def bad(*args, **kwargs):
        raise Exception("boom")


    monkeypatch.setattr(v, "ModelSpecs", bad)

    from ml_service.backend.configs.modeling.models.configs import RawConfigsWithLineage

    raw = RawConfigsWithLineage(model_specs={}, search={}, training={})
    with pytest.raises(ValueError) as exc:
        v.validate_all_configs(raw)

    assert "Config validation error" in str(exc.value)
    assert "boom" in str(exc.value)
