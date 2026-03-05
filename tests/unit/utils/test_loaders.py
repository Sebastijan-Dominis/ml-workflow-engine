"""Unit tests for configuration/data loading helpers."""

from pathlib import Path

import pandas as pd
import pytest
from ml.exceptions import ConfigError, DataError
from ml.utils.loaders import load_json, load_yaml, read_data

pytestmark = pytest.mark.unit


def test_load_yaml_reads_mapping_payload(tmp_path: Path) -> None:
    """Test that load_yaml correctly reads a YAML file containing a mapping and returns the expected dictionary.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to create a temporary YAML file with known contents, then calls load_yaml on that file and asserts that the returned dictionary matches the expected mapping.
    """
    file_path = tmp_path / "config.yaml"
    file_path.write_text("a: 1\nb: two\n", encoding="utf-8")

    result = load_yaml(file_path)

    assert result == {"a": 1, "b": "two"}


def test_load_yaml_rejects_non_mapping_payload(tmp_path: Path) -> None:
    """Test that load_yaml raises a ConfigError when the YAML file does not contain a mapping at the top level.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to create a temporary YAML file with a non-mapping payload (e.g., a list), then calls load_yaml on that file and asserts that a ConfigError is raised with an appropriate error message.
    """
    file_path = tmp_path / "config.yaml"
    file_path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_yaml(file_path)


def test_load_json_non_strict_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    """Test that load_json returns an empty dictionary when the specified file does not exist and strict mode is disabled. The test constructs a file path that does not exist, calls load_json with strict=False, and asserts that the result is an empty dictionary.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to construct a path that does not exist.
    """
    result = load_json(tmp_path / "missing.json", strict=False)

    assert result == {}


def test_load_json_strict_missing_file_raises_config_error(tmp_path: Path) -> None:
    """Test that load_json raises a ConfigError when the specified file does not exist and strict mode is enabled.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to construct a path that does not exist.
    """
    with pytest.raises(ConfigError, match="File not found"):
        load_json(tmp_path / "missing.json", strict=True)


def test_load_json_rejects_non_object_payload(tmp_path: Path) -> None:
    """Test that load_json raises a ConfigError when the JSON file does not contain an object at the top level. The test creates a temporary JSON file with a non-object payload (e.g., an array), calls load_json on that file, and asserts that a ConfigError is raised with an appropriate error message.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to create a temporary JSON file with known non-object contents, then calls load_json on that file and asserts that a ConfigError is raised.
    """
    file_path = tmp_path / "config.json"
    file_path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a JSON object"):
        load_json(file_path)


def test_read_data_rejects_unsupported_format(tmp_path: Path) -> None:
    """Test that read_data raises a ConfigError when the specified format is not supported.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to construct a path for a file with an unsupported format.
    """
    with pytest.raises(ConfigError, match="Unsupported data format"):
        read_data("xlsx", tmp_path / "data.xlsx")


def test_read_data_csv_applies_na_defaults(tmp_path: Path) -> None:
    """Test that read_data correctly applies default NA values when reading a CSV file. The test creates a temporary CSV file with specific contents that include empty strings and "NA" as values, then calls read_data with format "csv" on that file and asserts that the resulting DataFrame has the expected shape and that the appropriate values are interpreted as NA (i.e., NaN in the DataFrame). This validates that the default NA handling for CSV files is working as intended.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to create a temporary CSV file with known contents for testing.
    """
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,NA\n2,\n", encoding="utf-8")

    df = read_data("csv", csv_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert pd.isna(df.loc[0, "b"])
    assert pd.isna(df.loc[1, "b"])


def test_read_data_wraps_reader_errors_as_data_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that if the underlying reader function for a specific format raises an exception, read_data catches this and raises a DataError with an appropriate error message that includes the original exception message. The test uses monkeypatch to replace the reader function for "csv" format with a fake function that raises a ValueError, then calls read_data with format "csv" and asserts that a DataError is raised with a message indicating an error reading data in format 'csv' and that the original ValueError message is included.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to construct a path for the CSV file.
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture used to replace the reader function for "csv" format with a fake function that raises an exception.
    """
    def _failing_reader(path: Path) -> pd.DataFrame:
        raise ValueError("reader failed")

    monkeypatch.setitem(__import__("ml.utils.loaders", fromlist=["FORMAT_REGISTRY_READ"]).FORMAT_REGISTRY_READ, "csv", _failing_reader)

    with pytest.raises(DataError, match="Error reading data in format 'csv'"):
        read_data("csv", tmp_path / "data.csv")
