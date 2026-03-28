import argparse

import pipelines.features.freeze as freeze_mod
import yaml


def test_load_feature_registry_reads_yaml(tmp_path, monkeypatch):
    # create configs/feature_registry/features.yaml under tmp_path
    cfg_dir = tmp_path / "configs" / "feature_registry"
    cfg_dir.mkdir(parents=True)
    data = {"myset": {"v1": {"foo": "bar", "feature_store_path": "."}}}
    (cfg_dir / "features.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    res = freeze_mod.load_feature_registry("myset", "v1")
    assert res["foo"] == "bar"


def test_main_returns_exit_code_on_registry_load_failure(monkeypatch):
    monkeypatch.setattr(freeze_mod, "parse_args", lambda: argparse.Namespace(feature_set="fs", version="v", snapshot_binding_key=None, owner="o", logging_level="INFO"))
    monkeypatch.setattr(freeze_mod, "load_feature_registry", lambda f, v: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(freeze_mod, "resolve_exit_code", lambda e: 42)

    rc = freeze_mod.main()
    assert rc == 42
