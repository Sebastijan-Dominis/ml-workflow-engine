"""Integration tests for small formatting and environment helpers."""

from __future__ import annotations

import hashlib
from datetime import datetime

import pytest
from ml.exceptions import UserError
from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.formatting.str_to_bool import str_to_bool


def test_str_to_bool_variants_and_bool() -> None:
    assert str_to_bool(True) is True
    assert str_to_bool(False) is False
    assert str_to_bool("yes") is True
    assert str_to_bool("No") is False
    assert str_to_bool("1") is True
    assert str_to_bool("0") is False


def test_str_to_bool_invalid_raises() -> None:
    with pytest.raises(UserError):
        str_to_bool("maybe")


def test_iso_no_colon_formats_datetime() -> None:
    dt = datetime(2026, 3, 30, 12, 34, 56)
    s = iso_no_colon(dt)
    assert ":" not in s
    assert s.startswith("2026-03-30T12-34-56")


def test_parse_cuda_driver_version_if_pynvml_available() -> None:
    pytest.importorskip("pynvml")
    from ml.utils.runtime.gpu_info import parse_cuda_driver_version

    assert parse_cuda_driver_version(11040) == "11.4"
    assert parse_cuda_driver_version(10000) == "10.0"


def test_hash_environment_if_pynvml_available() -> None:
    pytest.importorskip("pynvml")
    from ml.utils.runtime.runtime_snapshot import hash_environment

    payload = "name: test\ndependencies:\n - python=3.10\n"
    expect = hashlib.sha256(payload.encode()).hexdigest()
    assert hash_environment(payload) == expect
