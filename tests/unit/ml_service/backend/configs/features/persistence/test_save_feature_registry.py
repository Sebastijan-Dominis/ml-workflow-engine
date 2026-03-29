import importlib


def test_save_feature_registry_creates_new_entry(tmp_path, monkeypatch):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.persistence.save_feature_registry"
    )

    captured = {}

    def fake_load_registry(path):
        return {}

    def fake_save_config(cfg, path):
        captured["cfg"] = cfg
        captured["path"] = path

    monkeypatch.setattr(mod, "load_registry", fake_load_registry)
    monkeypatch.setattr(mod, "save_config", fake_save_config)

    class DummyConfig:
        def model_dump(self, mode=None):
            return {"fields": []}

    out = mod.save_feature_registry(
        "featX",
        "v1",
        validated_config=DummyConfig(),
        registry_path=tmp_path / "cfgs" / "registry.yaml",
    )

    assert captured["cfg"] == {"featX": {"v1": {"fields": []}}}
    assert str(captured["path"]) == str(tmp_path / "cfgs" / "registry.yaml")
    assert out["status"] == "written"


def test_save_feature_registry_appends_to_existing(tmp_path, monkeypatch):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.persistence.save_feature_registry"
    )

    registry = {"featA": {"v1": {"old": 1}}}
    captured = {}

    monkeypatch.setattr(mod, "load_registry", lambda p: registry)

    def fake_save(cfg, path):
        captured["cfg"] = cfg

    monkeypatch.setattr(mod, "save_config", fake_save)

    class DummyConfig2:
        def model_dump(self, mode=None):
            return {"new": True}

    mod.save_feature_registry(
        "featA",
        "v2",
        validated_config=DummyConfig2(),
        registry_path=tmp_path / "registry.yaml",
    )

    assert "featA" in captured["cfg"]
    assert "v1" in captured["cfg"]["featA"]
    assert "v2" in captured["cfg"]["featA"]
    assert captured["cfg"]["featA"]["v1"] == {"old": 1}
    assert captured["cfg"]["featA"]["v2"] == {"new": True}


def test_model_dump_called_with_mode(monkeypatch, tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.persistence.save_feature_registry"
    )

    monkeypatch.setattr(mod, "load_registry", lambda p: {})
    monkeypatch.setattr(mod, "save_config", lambda cfg, p: None)

    called = {}

    class SpyConfig:
        def model_dump(self, mode=None):
            called["mode"] = mode
            return {"x": 1}

    mod.save_feature_registry(
        "f",
        "v",
        validated_config=SpyConfig(),
        registry_path=tmp_path / "r.yaml",
    )

    assert called.get("mode") == "json"
