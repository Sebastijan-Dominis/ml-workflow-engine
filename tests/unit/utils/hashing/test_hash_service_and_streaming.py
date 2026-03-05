"""Unit tests for the hash_streaming function in ml.utils.hashing.hash_streaming, as well as the wrapper functions in ml.utils.hashing.service that delegate to hash_streaming. The tests verify that hash_streaming produces consistent hashes for the same file contents, raises an error for missing files, and that the service wrapper functions correctly delegate to hash_streaming. Additionally, there is a test for the hash_thresholds function to ensure it produces a stable hash regardless of key order in the input dictionary."""
from pathlib import Path

import pytest
from ml.utils.hashing.hash_streaming import hash_streaming
from ml.utils.hashing.service import (
    hash_artifact,
    hash_data,
    hash_file,
    hash_thresholds,
)

pytestmark = pytest.mark.unit


def test_hash_streaming_returns_same_digest_for_same_file_contents(tmp_path: Path) -> None:
    """Test that hash_streaming produces the same digest for the same file contents, even when the file is read multiple times. The test creates a temporary file with specific contents, calls hash_streaming on it twice, and asserts that the resulting digests are the same and have the expected length.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in.
    """
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hotel-management\nquality-tests\n", encoding="utf-8")

    digest_1 = hash_streaming(file_path)
    digest_2 = hash_streaming(file_path)

    assert digest_1 == digest_2
    assert len(digest_1) == 64


def test_hash_streaming_raises_runtime_error_for_missing_file(tmp_path: Path) -> None:
    """Test that hash_streaming raises a RuntimeError when the specified file does not exist. The test constructs a file path that does not exist and asserts that calling hash_streaming on it raises the expected error with a message indicating an error hashing the file.

    Args:
        tmp_path (Path): The pytest fixture that provides a temporary directory for the test to create files in. The test uses this to construct a path that does not exist.
    """
    missing_path = tmp_path / "does_not_exist.txt"

    with pytest.raises(RuntimeError, match="Error hashing file"):
        hash_streaming(missing_path)


def test_hash_service_wrappers_delegate_to_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the hash service wrapper functions (hash_file, hash_data, hash_artifact) correctly delegate to the hash_streaming function. The test uses monkeypatch to replace hash_streaming with a fake function that records the file paths it is called with and returns a fixed digest. The test then calls each of the service wrapper functions with different file paths and asserts that they all return the expected digest and that hash_streaming was called with the correct file paths.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture used to replace hash_streaming with a fake function.
    """
    calls: list[Path] = []

    def _fake_hash_streaming(file_path: Path) -> str:
        calls.append(file_path)
        return "digest"

    monkeypatch.setattr("ml.utils.hashing.service.hash_streaming", _fake_hash_streaming)

    path_a = Path("a.csv")
    path_b = Path("b.parquet")
    path_c = Path("c.json")

    assert hash_file(path_a) == "digest"
    assert hash_data(path_b) == "digest"
    assert hash_artifact(path_c) == "digest"
    assert calls == [path_a, path_b, path_c]


def test_hash_thresholds_is_stable_for_reordered_keys() -> None:
    """Test that hash_thresholds produces the same hash for two dictionaries with the same content but different key orders. The test defines two dictionaries with the same nested content but different key orders, calls hash_thresholds on both, and asserts that the resulting hashes are the same.
    """
    payload_a = {"val": {"f1": 0.7, "roc_auc": 0.8}, "test": {"f1": 0.69, "roc_auc": 0.79}}
    payload_b = {"test": {"roc_auc": 0.79, "f1": 0.69}, "val": {"roc_auc": 0.8, "f1": 0.7}}

    assert hash_thresholds(payload_a) == hash_thresholds(payload_b)
