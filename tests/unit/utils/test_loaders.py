"""Unit tests for configuration/data loading helpers."""

from pathlib import Path

import pandas as pd
import pytest
from ml.exceptions import ConfigError, DataError
from ml.utils.loaders import load_json, load_yaml, read_data

pytestmark = pytest.mark.unit


def test_load_yaml_reads_mapping_payload(tmp_path: Path) -> None:
    """Verify that `load_yaml` returns a mapping payload."""
    file_path = tmp_path / "config.yaml"
    file_path.write_text("a: 1\nb: two\n", encoding="utf-8")

    result = load_yaml(file_path)

    assert result == {"a": 1, "b": "two"}


def test_load_yaml_rejects_non_mapping_payload(tmp_path: Path) -> None:
    """Verify that `load_yaml` rejects non-mapping YAML payloads."""
    file_path = tmp_path / "config.yaml"
    file_path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_yaml(file_path)


def test_load_json_non_strict_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    """Verify that non-strict `load_json` returns an empty dict for missing files."""
    result = load_json(tmp_path / "missing.json", strict=False)

    assert result == {}


def test_load_json_strict_missing_file_raises_config_error(tmp_path: Path) -> None:
    """Verify that strict `load_json` raises `ConfigError` for missing files."""
    with pytest.raises(ConfigError, match="File not found"):
        load_json(tmp_path / "missing.json", strict=True)


def test_load_json_rejects_non_object_payload(tmp_path: Path) -> None:
    """Verify that `load_json` rejects non-object JSON payloads."""
    file_path = tmp_path / "config.json"
    file_path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a JSON object"):
        load_json(file_path)


def test_read_data_rejects_unsupported_format(tmp_path: Path) -> None:
    """Verify that `read_data` rejects unsupported formats."""
    with pytest.raises(ConfigError, match="Unsupported data format"):
        read_data("xlsx", tmp_path / "data.xlsx")


def test_read_data_csv_applies_na_defaults(tmp_path: Path) -> None:
    """Verify that CSV reads apply default NA handling."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,NA\n2,\n", encoding="utf-8")

    df = read_data("csv", csv_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert pd.isna(df.loc[0, "b"])
    assert pd.isna(df.loc[1, "b"])


def test_read_data_wraps_reader_errors_as_data_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that reader failures are wrapped as `DataError`."""
    def _failing_reader(path: Path) -> pd.DataFrame:
        raise ValueError("reader failed")

    monkeypatch.setitem(__import__("ml.utils.loaders", fromlist=["FORMAT_REGISTRY_READ"]).FORMAT_REGISTRY_READ, "csv", _failing_reader)

    with pytest.raises(DataError, match="Error reading data in format 'csv'"):
        read_data("csv", tmp_path / "data.csv")
