"""Tests for snapshot_binding_key branch in ml.feature_freezing.utils.data_loader.

These tests focus on the code path that resolves dataset snapshots from a
snapshot binding key and ensures lineage entries are produced correctly.
"""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from ml.feature_freezing.freeze_strategies.tabular.config.models import (
    ConstraintsConfig,
    DatasetConfig,
    FeatureRolesConfig,
    LineageConfig,
    StorageConfig,
    TabularFeaturesConfig,
)


def test_load_data_with_lineage_uses_snapshot_binding(tmp_path, monkeypatch):
    """Ensure `load_data_with_lineage` uses `snapshot_binding_key` to resolve
    snapshots and produces correct lineage metadata and merged data.

    This test does not touch repository source code; it constructs a small
    dataset on-disk, patches external dependencies (registry, readers and
    merge helper), and asserts the returned lineage contains the expected
    snapshot id and loader/data hashes.
    """

    # Prepare on-disk dataset structure: <tmp>/<name>/<version>/<snapshot>/data.csv
    repo = tmp_path
    ds_name = "ds1"
    ds_version = "v1"
    snapshot_id = "snap123"
    path_suffix = "data.{format}"

    snapshot_dir = repo / ds_name / ds_version / snapshot_id
    snapshot_dir.mkdir(parents=True)
    dataset_file = snapshot_dir / path_suffix.format(format="csv")
    dataset_file.write_text("id,val\n1,10\n2,20")

    # Construct pydantic config models consumed by the loader (typed)
    dataset_cfg = DatasetConfig(
        ref=str(repo),
        name=ds_name,
        version=ds_version,
        format="csv",
        merge_key="id",
        path_suffix=path_suffix,
    )

    config = TabularFeaturesConfig(
        data=[dataset_cfg],
        feature_store_path=Path(str(repo)),
        columns=["id", "val"],
        feature_roles=FeatureRolesConfig(categorical=["id"], numerical=["val"], datetime=[]),
        constraints=ConstraintsConfig(forbid_nulls=[], max_cardinality={}),
        storage=StorageConfig(format="parquet", compression="snappy"),
        lineage=LineageConfig(created_by="test", created_at=datetime.utcnow()),
    )

    # Patch snapshot binding lookup to return mapping for our dataset
    monkeypatch.setattr(
        "ml.feature_freezing.utils.data_loader.get_and_validate_snapshot_binding",
        lambda key, expect_dataset_bindings=True: SimpleNamespace(
            datasets={ds_name: {ds_version: SimpleNamespace(snapshot=snapshot_id)}}
        ),
    )

    # Import the module under test and patch its IO/merge/hash dependencies
    import ml.feature_freezing.utils.data_loader as dl_mod

    # Ensure loader registry contains our format
    monkeypatch.setitem(dl_mod.HASH_LOADER_REGISTRY, "csv", lambda p: "loaderhash")

    # read_data should return a small DataFrame matching the on-disk CSV
    monkeypatch.setattr(dl_mod, "read_data", lambda fmt, path: pd.read_csv(path))

    # merge_dataset_into_main should return the df as merged data and a stable hash
    monkeypatch.setattr(
        dl_mod,
        "merge_dataset_into_main",
        lambda data, df, merge_key, dataset_name, dataset_version, dataset_snapshot_path, dataset_path: (df, "datahash"),
    )

    # Run the function under test
    data, lineage = dl_mod.load_data_with_lineage(config=config, snapshot_binding_key="any-key")

    assert len(lineage) == 1
    entry = lineage[0]
    assert entry.snapshot_id == snapshot_id
    assert entry.loader_validation_hash == "loaderhash"
    assert entry.data_hash == "datahash"
    assert data.shape[0] == 2
