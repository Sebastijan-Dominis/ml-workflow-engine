"""Unit tests for dataframe persistence helper used in data stages."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from ml.data.config.schemas.interim import InterimConfig
from ml.data.config.schemas.processed import ProcessedConfig
from ml.data.utils.persistence.save_data import save_data
from ml.exceptions import ConfigError, PersistenceError

pytestmark = pytest.mark.unit

StageConfig = InterimConfig | ProcessedConfig


def _config_stub(*, fmt: str = "parquet", suffix: str = "dataset.parquet", compression: str = "snappy") -> SimpleNamespace:
    """Create a minimal config-like object exposing save_data access attributes."""
    return SimpleNamespace(
        data=SimpleNamespace(
            output=SimpleNamespace(
                format=fmt,
                path_suffix=suffix,
                compression=compression,
            )
        )
    )


def _as_stage_config(config: SimpleNamespace) -> StageConfig:
    """Cast lightweight config stub to save_data function's expected schema union."""
    return cast(StageConfig, config)


def test_save_data_raises_config_error_for_unsupported_output_format(tmp_path: Path) -> None:
    """Reject persistence requests for formats missing from SAVE_FORMAT registry."""
    df = pd.DataFrame({"a": [1]})
    cfg = _config_stub(fmt="csv")

    with pytest.raises(ConfigError, match="Unsupported output format"):
        save_data(df, config=_as_stage_config(cfg), data_dir=tmp_path)


def test_save_data_creates_directory_and_returns_expected_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create destination directory and return file path from successful write."""
    df = pd.DataFrame({"a": [1]})
    cfg = _config_stub(suffix="outputs/data.parquet")
    target_dir = tmp_path / "nested" / "run"

    captured: dict[str, object] = {}

    def _fake_to_parquet(self: pd.DataFrame, path: Path, compression: str | None = None) -> None:
        captured["path"] = path
        captured["compression"] = compression

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)
    save_data_module = __import__(
        "ml.data.utils.persistence.save_data",
        fromlist=["SAVE_FORMAT"],
    )
    monkeypatch.setitem(save_data_module.SAVE_FORMAT, "parquet", pd.DataFrame.to_parquet)

    out_path = save_data(df, config=_as_stage_config(cfg), data_dir=target_dir)

    assert target_dir.exists()
    assert out_path == target_dir / "outputs/data.parquet"
    assert captured["path"] == target_dir / "outputs/data.parquet"
    assert captured["compression"] == "snappy"


def test_save_data_wraps_writer_failures_as_persistence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap storage backend failures in PersistenceError with save context."""
    df = pd.DataFrame({"a": [1]})
    cfg = _config_stub()

    def _raise(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise)
    save_data_module = __import__(
        "ml.data.utils.persistence.save_data",
        fromlist=["SAVE_FORMAT"],
    )
    monkeypatch.setitem(save_data_module.SAVE_FORMAT, "parquet", pd.DataFrame.to_parquet)

    with pytest.raises(PersistenceError, match="Error saving data"):
        save_data(df, config=_as_stage_config(cfg), data_dir=tmp_path)
