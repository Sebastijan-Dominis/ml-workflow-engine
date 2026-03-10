"""Unit tests for feature snapshot resolution helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from ml.exceptions import DataError
from ml.features.loading.resolve_feature_snapshots import resolve_feature_snapshots
from ml.modeling.models.feature_lineage import FeatureLineage

pytestmark = pytest.mark.unit


def _lineage(snapshot_id: str) -> FeatureLineage:
    """Build a minimal valid lineage entry for snapshot-binding tests."""
    return FeatureLineage(
        name="booking_context_features",
        version="v1",
        snapshot_id=snapshot_id,
        file_hash="file-hash",
        in_memory_hash="mem-hash",
        feature_schema_hash="schema-hash",
        operator_hash="op-hash",
        feature_type="tabular",
    )


def test_resolve_feature_snapshots_uses_latest_snapshot_when_binding_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve snapshot via latest-snapshot helper when explicit binding is not provided."""
    snapshot_path = tmp_path / "feature_store" / "booking_context_features" / "v1" / "snap_001"
    snapshot_path.mkdir(parents=True)
    (snapshot_path / "metadata.json").write_text("{}", encoding="utf-8")

    observed_metadata_paths: list[Path] = []

    monkeypatch.setattr(
        "ml.features.loading.resolve_feature_snapshots.get_latest_snapshot_path",
        lambda _: snapshot_path,
    )

    def _fake_load_json(path: Path) -> dict[str, str]:
        observed_metadata_paths.append(path)
        return {"schema": "ok"}

    monkeypatch.setattr("ml.features.loading.resolve_feature_snapshots.load_json", _fake_load_json)

    resolved = resolve_feature_snapshots(
        feature_store_path=tmp_path / "feature_store",
        feature_sets=[SimpleNamespace(name="booking_context_features", version="v1")],
    )

    assert len(resolved) == 1
    assert resolved[0]["snapshot_path"] == snapshot_path
    assert resolved[0]["snapshot_id"] == "snap_001"
    assert resolved[0]["metadata"] == {"schema": "ok"}
    assert observed_metadata_paths == [snapshot_path / "metadata.json"]


def test_resolve_feature_snapshots_uses_snapshot_binding_and_skips_latest_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer bound snapshot identifiers and avoid calling latest-snapshot resolution."""
    snapshot_path = tmp_path / "feature_store" / "booking_context_features" / "v1" / "bound_snap"
    snapshot_path.mkdir(parents=True)
    (snapshot_path / "metadata.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "ml.features.loading.resolve_feature_snapshots.get_latest_snapshot_path",
        lambda _: (_ for _ in ()).throw(AssertionError("latest lookup must not run when binding is provided")),
    )
    monkeypatch.setattr(
        "ml.features.loading.resolve_feature_snapshots.load_json",
        lambda _: {"bound": True},
    )

    resolved = resolve_feature_snapshots(
        feature_store_path=tmp_path / "feature_store",
        feature_sets=[SimpleNamespace(name="booking_context_features", version="v1")],
        snapshot_binding=[_lineage("bound_snap")],
    )

    assert resolved[0]["snapshot_path"] == snapshot_path
    assert resolved[0]["snapshot_id"] == "bound_snap"
    assert resolved[0]["metadata"] == {"bound": True}


def test_resolve_feature_snapshots_raises_for_missing_metadata_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise ``DataError`` when resolved snapshot directory lacks ``metadata.json``."""
    snapshot_path = tmp_path / "feature_store" / "booking_context_features" / "v1" / "snap_without_metadata"
    snapshot_path.mkdir(parents=True)

    monkeypatch.setattr(
        "ml.features.loading.resolve_feature_snapshots.get_latest_snapshot_path",
        lambda _: snapshot_path,
    )

    with pytest.raises(DataError, match="File not found"):
        resolve_feature_snapshots(
            feature_store_path=tmp_path / "feature_store",
            feature_sets=[SimpleNamespace(name="booking_context_features", version="v1")],
        )


def test_resolve_feature_snapshots_respects_feature_set_order_for_multiple_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return resolved descriptors in the same order as incoming feature-set specs."""
    fs_root = tmp_path / "feature_store"
    snap_a = fs_root / "a_features" / "v1" / "snap_a"
    snap_b = fs_root / "b_features" / "v2" / "snap_b"
    snap_a.mkdir(parents=True)
    snap_b.mkdir(parents=True)
    (snap_a / "metadata.json").write_text("{}", encoding="utf-8")
    (snap_b / "metadata.json").write_text("{}", encoding="utf-8")

    snapshots = [snap_a, snap_b]
    monkeypatch.setattr(
        "ml.features.loading.resolve_feature_snapshots.get_latest_snapshot_path",
        lambda _: snapshots.pop(0),
    )
    monkeypatch.setattr("ml.features.loading.resolve_feature_snapshots.load_json", lambda _: {"ok": 1})

    resolved = resolve_feature_snapshots(
        feature_store_path=fs_root,
        feature_sets=[
            SimpleNamespace(name="a_features", version="v1"),
            SimpleNamespace(name="b_features", version="v2"),
        ],
    )

    assert [item["snapshot_id"] for item in resolved] == ["snap_a", "snap_b"]
    assert [item["fs_spec"].name for item in resolved] == ["a_features", "b_features"]
