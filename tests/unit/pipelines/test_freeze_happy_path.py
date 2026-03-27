from types import SimpleNamespace

import pipelines.features.freeze as freeze_mod
import yaml


def test_freeze_main_happy_path(tmp_path, monkeypatch):
    # prepare fake config file
    cfg_dir = tmp_path / "configs" / "feature_registry"
    cfg_dir.mkdir(parents=True)
    data = {"myset": {"v1": {"feature_store_path": str(tmp_path)}}}
    (cfg_dir / "features.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    # ensure working dir is tmp_path so load_feature_registry reads our file
    monkeypatch.chdir(tmp_path)

    # patch CLI args
    monkeypatch.setattr(freeze_mod, "parse_args", lambda: SimpleNamespace(feature_set="myset", version="v1", snapshot_binding_key=None, owner="me", logging_level="INFO"))

    # patch functions that perform heavy side effects
    monkeypatch.setattr(freeze_mod, "get_strategy_type", lambda raw: "tabular")
    monkeypatch.setattr(freeze_mod, "validate_feature_registry", lambda raw, t: SimpleNamespace(feature_store_path=str(tmp_path), type="tabular"))
    monkeypatch.setattr(freeze_mod, "bootstrap_logging", lambda **k: None)
    monkeypatch.setattr(freeze_mod, "add_file_handler", lambda *a, **k: None)

    class FakeStrategy:
        def freeze(self, config, snapshot_binding_key, snapshot_id, timestamp, start_time, owner):
            out_dir = tmp_path / snapshot_id
            out_dir.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(snapshot_path=out_dir, metadata={"snapshot_id": snapshot_id})

    monkeypatch.setattr(freeze_mod, "get_strategy", lambda t: FakeStrategy())
    monkeypatch.setattr(freeze_mod, "save_metadata", lambda metadata, target_dir: None)

    rc = freeze_mod.main()
    assert rc == 0
