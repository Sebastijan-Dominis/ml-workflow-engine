"""Unit tests for promotion registry update and diff persistence helpers."""

from pathlib import Path

import pytest
import yaml
from ml.exceptions import PersistenceError
from ml.promotion.persistence.registry import persist_registry_diff, update_registry_and_archive

pytestmark = pytest.mark.unit


def test_update_registry_and_archive_production_archives_previous_and_updates_registry(tmp_path: Path) -> None:
    """Archive previous production entry and atomically write updated registry for production promotions."""
    registry_path = tmp_path / "models.yaml"
    archive_path = tmp_path / "archive.yaml"

    model_registry = {
        "cancellation": {
            "city_hotel": {
                "production": {
                    "promotion_id": "prom-old",
                    "metrics": {"val": {"f1": 0.79}},
                }
            }
        }
    }
    archive_registry = {"cancellation": {"city_hotel": {}}}
    def _make_entry(promotion_id: str, metrics: dict | None = None) -> dict:
        return {
            "experiment_id": "exp-1",
            "train_run_id": "train-1",
            "eval_run_id": "eval-1",
            "explain_run_id": "explain-1",
            "model_version": "v1",
            "artifacts": {"model_hash": "h1", "model_path": "s3://m"},
            "feature_lineage": [
                {
                    "name": "fs",
                    "version": "v1",
                    "snapshot_id": "s1",
                    "file_hash": "f1",
                    "in_memory_hash": "im1",
                    "feature_schema_hash": "sch1",
                    "operator_hash": "op1",
                    "feature_type": "tabular",
                    "file_name": "features.py",
                    "data_format": "parquet",
                }
            ],
            "metrics": metrics or {"train": {"f1": 0.82}, "val": {"f1": 0.82}, "test": {"f1": 0.82}},
            "git_commit": "abc",
            "promotion_id": promotion_id,
            "promoted_at": "2026-01-01T00:00:00Z",
        }

    run_info = _make_entry("prom-new")

    updated = update_registry_and_archive(
        model_registry=model_registry,
        archive_registry=archive_registry,
        stage="production",
        run_info=run_info,
        problem="cancellation",
        segment="city_hotel",
        registry_path=registry_path,
        archive_path=archive_path,
    )

    assert model_registry["cancellation"]["city_hotel"]["production"]["promotion_id"] == "prom-old"
    assert updated["cancellation"]["city_hotel"]["production"] == run_info

    saved_registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    saved_archive = yaml.safe_load(archive_path.read_text(encoding="utf-8"))

    assert saved_registry == updated
    assert saved_archive["cancellation"]["city_hotel"]["prom-old"]["promotion_id"] == "prom-old"


def test_update_registry_and_archive_production_raises_when_previous_missing_promotion_id(tmp_path: Path) -> None:
    """Raise PersistenceError when existing production entry cannot be archived due to missing promotion id."""
    registry_path = tmp_path / "models.yaml"
    archive_path = tmp_path / "archive.yaml"

    # supply a full, valid registry entry payload so validation passes
    valid_run = {
        "experiment_id": "exp-2",
        "train_run_id": "train-2",
        "eval_run_id": "eval-2",
        "explain_run_id": "explain-2",
        "model_version": "v2",
        "artifacts": {"model_hash": "h2", "model_path": "s3://m2"},
        "feature_lineage": [
            {
                "name": "fs2",
                "version": "v2",
                "snapshot_id": "s2",
                "file_hash": "f2",
                "in_memory_hash": "im2",
                "feature_schema_hash": "sch2",
                "operator_hash": "op2",
                    "feature_type": "tabular",
                    "file_name": "features.py",
                    "data_format": "parquet",
            }
        ],
        "metrics": {"train": {"f1": 0.79}, "val": {"f1": 0.79}, "test": {"f1": 0.79}},
        "git_commit": "def",
        "promotion_id": "prom-new",
        "promoted_at": "2026-01-02T00:00:00Z",
    }

    with pytest.raises(PersistenceError, match="missing promotion_id"):
        update_registry_and_archive(
            model_registry={"cancellation": {"city_hotel": {"production": {"metrics": {"val": {"f1": 0.79}}}}}},
            archive_registry={},
            stage="production",
            run_info=valid_run,
            problem="cancellation",
            segment="city_hotel",
            registry_path=registry_path,
            archive_path=archive_path,
        )


def test_update_registry_and_archive_staging_updates_staging_without_archiving(tmp_path: Path) -> None:
    """Write staging entry directly without touching archive file for staging promotions."""
    registry_path = tmp_path / "models.yaml"
    archive_path = tmp_path / "archive.yaml"

    valid_staging = {
        "experiment_id": "exp-stage",
        "train_run_id": "train-stage",
        "eval_run_id": "eval-stage",
        "explain_run_id": "explain-stage",
        "model_version": "v-stage",
        "artifacts": {"model_hash": "hs", "model_path": "s3://ms"},
        "feature_lineage": [
            {
                "name": "fs-s",
                "version": "vs",
                "snapshot_id": "ss",
                "file_hash": "fs",
                "in_memory_hash": "ims",
                "feature_schema_hash": "schs",
                "operator_hash": "ops",
                "feature_type": "tabular",
                "file_name": "features.py",
                "data_format": "parquet",
            }
        ],
        "metrics": {"train": {"f1": 0.5}, "val": {"f1": 0.5}, "test": {"f1": 0.5}},
        "git_commit": "ghi",
        "promotion_id": "prom-stage",
        "promoted_at": "2026-01-03T00:00:00Z",
    }

    updated = update_registry_and_archive(
        model_registry={"cancellation": {"city_hotel": {"production": {"promotion_id": "prom-old"}}}},
        archive_registry={},
        stage="staging",
        run_info=valid_staging,
        problem="cancellation",
        segment="city_hotel",
        registry_path=registry_path,
        archive_path=archive_path,
    )

    assert updated["cancellation"]["city_hotel"]["staging"] == valid_staging
    assert not archive_path.exists()


def test_update_registry_and_archive_initializes_missing_problem_and_segment_keys(tmp_path: Path) -> None:
    """Create missing problem/segment structure and persist first production record."""
    registry_path = tmp_path / "models.yaml"
    archive_path = tmp_path / "archive.yaml"

    valid_first = {
        "experiment_id": "exp-first",
        "train_run_id": "t1",
        "eval_run_id": "e1",
        "explain_run_id": "x1",
        "model_version": "v-first",
        "artifacts": {"model_hash": "h-first", "model_path": "s3://first"},
        "feature_lineage": [
            {
                "name": "fsf",
                "version": "vf",
                "snapshot_id": "sf",
                "file_hash": "ff",
                "in_memory_hash": "imf",
                "feature_schema_hash": "schf",
                "operator_hash": "opf",
                "feature_type": "tabular",
                "file_name": "features.py",
                "data_format": "parquet",
            }
        ],
        "metrics": {"train": {"f1": 0.91}, "val": {"f1": 0.91}, "test": {"f1": 0.91}},
        "git_commit": "jkl",
        "promotion_id": "prom-first",
        "promoted_at": "2026-01-04T00:00:00Z",
    }

    updated = update_registry_and_archive(
        model_registry={},
        archive_registry={},
        stage="production",
        run_info=valid_first,
        problem="no_show",
        segment="global",
        registry_path=registry_path,
        archive_path=archive_path,
    )

    assert updated == {
        "no_show": {"global": {"production": valid_first}}
    }
    assert not archive_path.exists()


def test_update_registry_and_archive_wraps_filesystem_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Raise PersistenceError when registry/archive writes fail at filesystem boundary."""
    registry_path = tmp_path / "models.yaml"
    archive_path = tmp_path / "archive.yaml"

    monkeypatch.setattr("ml.promotion.persistence.registry.os.replace", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))

    valid_staging = {
        "experiment_id": "exp-disk",
        "train_run_id": "train-d",
        "eval_run_id": "eval-d",
        "explain_run_id": "explain-d",
        "model_version": "v-d",
        "artifacts": {"model_hash": "hd", "model_path": "s3://md"},
        "feature_lineage": [
            {
                "name": "fsd",
                "version": "vd",
                "snapshot_id": "sd",
                "file_hash": "fd",
                "in_memory_hash": "imd",
                "feature_schema_hash": "schd",
                "operator_hash": "opd",
                "feature_type": "tabular",
                "file_name": "features.py",
                "data_format": "parquet",
            }
        ],
        "metrics": {"train": {"f1": 0.1}, "val": {"f1": 0.1}, "test": {"f1": 0.1}},
        "git_commit": "mno",
        "promotion_id": "prom-d",
        "promoted_at": "2026-01-05T00:00:00Z",
    }

    with pytest.raises(PersistenceError, match="Failed to update model registry and archive"):
        update_registry_and_archive(
            model_registry={},
            archive_registry={},
            stage="staging",
            run_info=valid_staging,
            problem="cancellation",
            segment="city_hotel",
            registry_path=registry_path,
            archive_path=archive_path,
        )


def test_persist_registry_diff_writes_previous_and_updated_sections(tmp_path: Path) -> None:
    """Persist registry diff file containing previous and updated registry snapshots."""
    previous = {"a": 1}
    updated = {"a": 2}

    persist_registry_diff(previous_registry=previous, updated_registry=updated, run_dir=tmp_path)

    diff_path = tmp_path / "registry_diff.yaml"
    payload = yaml.safe_load(diff_path.read_text(encoding="utf-8"))
    assert payload == {"previous": previous, "updated": updated}


def test_persist_registry_diff_wraps_write_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Raise PersistenceError when registry diff file cannot be written."""
    def _failing_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _failing_open)

    with pytest.raises(PersistenceError, match="Failed to persist registry diff"):
        persist_registry_diff(previous_registry={"a": 1}, updated_registry={"a": 2}, run_dir=tmp_path)
