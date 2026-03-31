"""Integration tests for `ml.utils.loaders` file loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from ml.exceptions import ConfigError, DataError
from ml.utils.loaders import load_json, load_yaml, read_data


def test_load_yaml_success_and_invalid(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("a: 1\nb: hello\n")
    got = load_yaml(p)
    assert got["a"] == 1
    assert got["b"] == "hello"

    # non-mapping YAML should raise
    p2 = tmp_path / "list.yaml"
    p2.write_text("- a\n- b\n")
    with pytest.raises(ConfigError):
        load_yaml(p2)

    # missing file raises
    with pytest.raises(ConfigError):
        load_yaml(tmp_path / "nope.yaml")


def test_load_json_strict_and_non_strict_and_invalid(tmp_path: Path) -> None:
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"k": "v"}))
    got = load_json(p)
    assert got == {"k": "v"}

    # missing strict -> DataError
    with pytest.raises(DataError):
        load_json(tmp_path / "missing.json", strict=True)

    # missing non-strict -> empty dict
    assert load_json(tmp_path / "missing.json", strict=False) == {}

    # invalid JSON -> ConfigError
    p2 = tmp_path / "bad.json"
    p2.write_text("{ not json }")
    with pytest.raises(ConfigError):
        load_json(p2)

    # non-object JSON (array) -> ConfigError
    p3 = tmp_path / "arr.json"
    p3.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ConfigError):
        load_json(p3)


def test_read_data_csv_and_unsupported(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n3,4\n")

    df = read_data("csv", csv)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert df.shape[0] == 2

    with pytest.raises(ConfigError):
        read_data("xml", csv)
