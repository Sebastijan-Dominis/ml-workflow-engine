import importlib


def test_load_feature_registry_missing(tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.loading.load_registry"
    )
    p = tmp_path / "nope.yaml"
    assert not p.exists()
    assert mod.load_feature_registry(p) == {}


def test_load_feature_registry_empty_file(tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.loading.load_registry"
    )
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert mod.load_feature_registry(p) == {}


def test_load_feature_registry_parses_yaml(tmp_path):
    mod = importlib.import_module(
        "ml_service.backend.configs.features.loading.load_registry"
    )
    p = tmp_path / "reg.yaml"
    p.write_text("foo:\n  bar: 1\n")
    res = mod.load_feature_registry(p)
    assert isinstance(res, dict)
    assert res == {"foo": {"bar": 1}}
