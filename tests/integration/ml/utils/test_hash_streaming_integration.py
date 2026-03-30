"""Integration tests for streaming file hashing utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ml.utils.hashing.hash_streaming import hash_streaming


def test_hash_streaming_matches_direct_hash(tmp_path: Path) -> None:
    p = tmp_path / "data.bin"
    content = b"hello world\n" * 10
    p.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    got = hash_streaming(p)
    assert got == expected


def test_hash_streaming_with_small_chunks(tmp_path: Path) -> None:
    p = tmp_path / "data2.bin"
    content = b"abc123" * 1000
    p.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    got = hash_streaming(p, chunk_size=16)
    assert got == expected
