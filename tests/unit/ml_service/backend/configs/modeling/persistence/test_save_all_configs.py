import importlib
from pathlib import Path


def test_save_all_configs_calls_save_config(monkeypatch, tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.modeling.persistence.save_all_configs"
    )

    calls = []

    def fake_save(data, path):
        calls.append((data, path))

    monkeypatch.setattr(mod, "save_config", fake_save)

    class FakeModel:
        def __init__(self, payload):
            self.payload = payload

        def model_dump(self, mode="json", exclude=None):
            d = dict(self.payload)
            if exclude:
                for k in list(exclude):
                    d.pop(k, None)
            return d

    # use typing.cast so static type checkers accept our test fakes
    from typing import cast

    from ml.config.schemas.model_specs import ModelSpecs
    from ml_service.backend.configs.modeling.models.configs import (
        ConfigPaths,
        SearchConfigForValidation,
        TrainConfigForValidation,
        ValidatedConfigs,
    )

    model_specs = cast(ModelSpecs, FakeModel({"foo": "bar", "meta": {"x": 1}}))
    search = cast(SearchConfigForValidation, FakeModel({"search": True}))
    training = cast(TrainConfigForValidation, FakeModel({"train": True}))

    from types import SimpleNamespace

    validated = cast(
        ValidatedConfigs,
        SimpleNamespace(model_specs=model_specs, search=search, training=training),
    )
    paths = ConfigPaths(
        model_specs=str(tmp_path / "m_specs.yaml"),
        search=str(tmp_path / "search.yaml"),
        training=str(tmp_path / "train.yaml"),
    )

    mod.save_all_configs(validated, paths)

    assert len(calls) == 3

    data0, path0 = calls[0]
    assert "meta" not in data0
    assert path0 == Path(paths.model_specs)

    assert calls[1][0] == {"search": True}
    assert calls[1][1] == Path(paths.search)

    assert calls[2][0] == {"train": True}
    assert calls[2][1] == Path(paths.training)
