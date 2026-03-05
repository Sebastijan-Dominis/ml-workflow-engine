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
    run_info = {"promotion_id": "prom-new", "metrics": {"val": {"f1": 0.82}}}

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

    with pytest.raises(PersistenceError, match="missing promotion_id"):
        update_registry_and_archive(
            model_registry={"cancellation": {"city_hotel": {"production": {"metrics": {"val": {"f1": 0.79}}}}}},
            archive_registry={},
            stage="production",
            run_info={"promotion_id": "prom-new"},
            problem="cancellation",
            segment="city_hotel",
            registry_path=registry_path,
            archive_path=archive_path,
        )


def test_update_registry_and_archive_staging_updates_staging_without_archiving(tmp_path: Path) -> None:
    """Write staging entry directly without touching archive file for staging promotions."""
    registry_path = tmp_path / "models.yaml"
    archive_path = tmp_path / "archive.yaml"

    updated = update_registry_and_archive(
        model_registry={"cancellation": {"city_hotel": {"production": {"promotion_id": "prom-old"}}}},
        archive_registry={},
        stage="staging",
        run_info={"staging_id": "stage-1"},
        problem="cancellation",
        segment="city_hotel",
        registry_path=registry_path,
        archive_path=archive_path,
    )

    assert updated["cancellation"]["city_hotel"]["staging"] == {"staging_id": "stage-1"}
    assert not archive_path.exists()


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
