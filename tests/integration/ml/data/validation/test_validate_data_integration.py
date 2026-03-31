from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ml.data.validation.validate_data import validate_data
from ml.exceptions import UserError

pytestmark = pytest.mark.integration


def test_validate_data_no_expected_hash_returns_empty(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    data_file.write_text("1,2,3")

    metadata: dict[str, Any] = {}

    res = validate_data(data_path=data_file, metadata=metadata)
    assert res == ""


def test_validate_data_mismatch_raises(monkeypatch: Any, tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    data_file.write_text("1,2,3")

    metadata: dict[str, Any] = {"data": {"hash": "expectedhash"}}

    # Patch the imported hash_data to return a different hash
    import ml.data.validation.validate_data as mod

    monkeypatch.setattr(mod, "hash_data", lambda p: "actualhash")

    with pytest.raises(UserError):
        validate_data(data_path=data_file, metadata=metadata)
