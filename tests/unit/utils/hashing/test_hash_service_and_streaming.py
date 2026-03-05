"""Unit tests for streaming hashing and hashing service wrappers."""
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
    """Verify deterministic digests for identical file contents."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hotel-management\nquality-tests\n", encoding="utf-8")

    digest_1 = hash_streaming(file_path)
    digest_2 = hash_streaming(file_path)

    assert digest_1 == digest_2
    assert len(digest_1) == 64


def test_hash_streaming_raises_runtime_error_for_missing_file(tmp_path: Path) -> None:
    """Verify missing files raise `RuntimeError` during hashing."""
    missing_path = tmp_path / "does_not_exist.txt"

    with pytest.raises(RuntimeError, match="Error hashing file"):
        hash_streaming(missing_path)


def test_hash_service_wrappers_delegate_to_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify wrapper helpers delegate directly to `hash_streaming`."""
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
    """Verify stable threshold hashes for equivalent payloads with reordered keys."""
    payload_a = {"val": {"f1": 0.7, "roc_auc": 0.8}, "test": {"f1": 0.69, "roc_auc": 0.79}}
    payload_b = {"test": {"roc_auc": 0.79, "f1": 0.69}, "val": {"roc_auc": 0.8, "f1": 0.7}}

    assert hash_thresholds(payload_a) == hash_thresholds(payload_b)
