import argparse
from pathlib import Path
from types import SimpleNamespace

import ml.post_promotion.shared.loading.features as feat_mod
import pandas as pd
from ml.promotion.config.registry_entry import (
    RegistryArtifacts,
    RegistryEntry,
    RegistryEntryMetrics,
    RegistryFeatureSetLineage,
)


def _build_registry_entry() -> RegistryEntry:
    artifacts = RegistryArtifacts(model_hash="h", model_path="p")
    feature_lineage = [
        RegistryFeatureSetLineage(
            name="fl",
            version="v1",
            snapshot_id="s1",
            file_hash="fh",
            in_memory_hash="imh",
            feature_schema_hash="fsh",
            operator_hash="oh",
            feature_type="tabular",
        )
    ]
    metrics = RegistryEntryMetrics(train={}, val={}, test={})
    return RegistryEntry(
        experiment_id="e",
        train_run_id="tr",
        eval_run_id="er",
        explain_run_id="xr",
        model_version="v",
        artifacts=artifacts,
        feature_lineage=feature_lineage,
        metrics=metrics,
        git_commit="gc",
    )


def test_prepare_features_default_snapshot_binding(monkeypatch, tmp_path):
    args = argparse.Namespace(problem="prob", segment="seg")
    model_meta = _build_registry_entry()

    # stub load_json to return minimal raw data
    monkeypatch.setattr(feat_mod, "load_json", lambda path: {})

    # stub validate_training_metadata to return an object with lineage.feature_lineage
    training_meta = SimpleNamespace(lineage=SimpleNamespace(feature_lineage=["fs"]))
    monkeypatch.setattr(feat_mod, "validate_training_metadata", lambda raw: training_meta)

    # capture resolved_snapshots returned and ensure passed through
    resolved = {"fs": "snap1"}
    def fake_resolve(feature_store_path, feature_sets, snapshot_binding=None, snapshot_binding_key=None):
        assert feature_store_path == Path("feature_store")
        # when default branch, snapshot_binding should be passed
        assert snapshot_binding == training_meta.lineage.feature_lineage
        return resolved

    monkeypatch.setattr(feat_mod, "resolve_feature_snapshots", fake_resolve)

    # stub validate_search_record to provide metadata.version and env
    monkeypatch.setattr(feat_mod, "validate_search_record", lambda raw: SimpleNamespace(metadata=SimpleNamespace(version="mv", env="dev")))

    # stub load_and_validate_config to return a dummy model_cfg
    monkeypatch.setattr(feat_mod, "load_and_validate_config", lambda path, search_dir, cfg_type, env: SimpleNamespace(cfg=True))

    # stub the lazy-loaded load_features_and_target to return controlled X, y, _, entity_key
    df = pd.DataFrame({"a": [1, 2]})
    ser = pd.Series([0, 1])
    import ml.features.loading.features_and_target as fat_mod
    monkeypatch.setattr(fat_mod, "load_features_and_target", lambda model_cfg, snapshot_selection, drop_entity_key, strict: (df, ser, None, "id"))

    res = feat_mod.prepare_features(args=args, model_metadata=model_meta)

    pd.testing.assert_frame_equal(res.features, df)
    pd.testing.assert_series_equal(res.target, ser)
    assert res.entity_key == "id"


def test_prepare_features_with_snapshot_bindings_key(monkeypatch):
    args = argparse.Namespace(problem="prob", segment="seg")
    model_meta = _build_registry_entry()

    monkeypatch.setattr(feat_mod, "load_json", lambda path: {})
    training_meta = SimpleNamespace(lineage=SimpleNamespace(feature_lineage=["fs"]))
    monkeypatch.setattr(feat_mod, "validate_training_metadata", lambda raw: training_meta)

    # ensure resolve_feature_snapshots is called with snapshot_binding_key when provided
    def fake_resolve(feature_store_path, feature_sets, snapshot_binding=None, snapshot_binding_key=None):
        assert snapshot_binding is None
        assert snapshot_binding_key == "key1"
        return {"fs": "snap1"}

    monkeypatch.setattr(feat_mod, "resolve_feature_snapshots", fake_resolve)
    monkeypatch.setattr(feat_mod, "validate_search_record", lambda raw: SimpleNamespace(metadata=SimpleNamespace(version="mv", env="dev")))
    monkeypatch.setattr(feat_mod, "load_and_validate_config", lambda path, search_dir, cfg_type, env: SimpleNamespace(cfg=True))

    import ml.features.loading.features_and_target as fat_mod
    df = pd.DataFrame({"b": [3]})
    ser = pd.Series([1])
    monkeypatch.setattr(fat_mod, "load_features_and_target", lambda model_cfg, snapshot_selection, drop_entity_key, strict: (df, ser, None, "ek"))

    res = feat_mod.prepare_features(args=args, model_metadata=model_meta, snapshot_bindings_id="key1")

    pd.testing.assert_frame_equal(res.features, df)
    pd.testing.assert_series_equal(res.target, ser)
    assert res.entity_key == "ek"
