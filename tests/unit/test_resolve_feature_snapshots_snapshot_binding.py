"""Tests for snapshot_binding_key branch in ml.features.loading.resolve_feature_snapshots.

These tests validate that when a `snapshot_binding_key` is provided the
resolver picks the configured snapshot id and returns the expected
snapshot path and metadata stub.
"""

from types import SimpleNamespace


def test_resolve_feature_snapshots_with_snapshot_binding_key(tmp_path, monkeypatch):
    """Verify `resolve_feature_snapshots` picks snapshots using a binding key.

    The test creates a minimal feature store layout on-disk and patches
    `get_and_validate_snapshot_binding` to return a binding that points to
    the created snapshot. `load_json` is patched to avoid depending on the
    real loader implementation.
    """

    fs_name = "fs1"
    fs_version = "v1"
    snapshot_id = "snapA"

    feature_store = tmp_path
    snapshot_dir = feature_store / fs_name / fs_version / snapshot_id
    snapshot_dir.mkdir(parents=True)

    # Patch get_and_validate_snapshot_binding to return a mapping for our feature set
    monkeypatch.setattr(
        "ml.features.loading.resolve_feature_snapshots.get_and_validate_snapshot_binding",
        lambda key, expect_feature_set_bindings=True: SimpleNamespace(
            feature_sets={fs_name: {fs_version: SimpleNamespace(snapshot=snapshot_id)}}
        ),
    )

    # Patch load_json to return controlled metadata without file IO
    import ml.features.loading.resolve_feature_snapshots as rf_mod
    monkeypatch.setattr(rf_mod, "load_json", lambda p: {"ok": True})

    fs = SimpleNamespace(name=fs_name, version=fs_version)
    resolved = rf_mod.resolve_feature_snapshots(
        feature_store_path=feature_store,
        feature_sets=[fs],
        snapshot_binding=None,
        snapshot_binding_key="key1",
    )

    assert len(resolved) == 1
    assert resolved[0]["snapshot_id"] == snapshot_id
    assert resolved[0]["snapshot_path"] == snapshot_dir
    assert resolved[0]["metadata"] == {"ok": True}
