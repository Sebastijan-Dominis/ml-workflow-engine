"""Unit tests for runtime-info validation wrapper in modeling validation module."""

from __future__ import annotations

import pytest
from ml.exceptions import RuntimeMLError
from ml.modeling.models.runtime_info import RuntimeInfo
from ml.modeling.validation.runtime_info import validate_runtime_info

pytestmark = pytest.mark.unit


def _runtime_info_payload() -> dict[str, object]:
    """Build a minimal valid runtime-info payload matching the RuntimeInfo model."""
    return {
        "environment": {
            "conda_env_export": "name: hotel_management",
            "conda_env_hash": "env-hash-123",
        },
        "execution": {
            "created_at": "2026-03-06T12:34:56Z",
            "duration_seconds": 12.5,
            "git_commit": "abc123",
            "python_executable": "C:/Users/Sebastijan/anaconda3/envs/hotel_management/python.exe",
        },
        "gpu_info": {
            "cuda_version": "12.1",
            "gpu_count": 1,
            "gpu_devices_available": [0],
            "gpu_devices_used": [0],
            "gpu_driver_version": "555.10",
            "gpu_memories_gb": [24.0],
            "gpu_names": ["NVIDIA RTX 4090"],
            "task_type": "GPU",
        },
        "runtime": {
            "os": "Windows",
            "os_release": "11",
            "architecture": "AMD64",
            "processor": "x86_64",
            "ram_total_gb": 64.0,
            "platform_string": "Windows-11-10.0.22631-SP0",
            "hostname": "dev-machine",
            "python_version": "3.11.14",
            "python_impl": "CPython",
            "python_build": ["main", "Feb 1 2026 00:00:00"],
        },
    }


def test_validate_runtime_info_returns_typed_model_for_valid_payload() -> None:
    """Return a ``RuntimeInfo`` model for valid nested runtime metadata payloads."""
    payload = _runtime_info_payload()

    result = validate_runtime_info(payload)

    assert isinstance(result, RuntimeInfo)
    assert result.execution.git_commit == "abc123"
    assert result.gpu_info.task_type.value == "GPU"


def test_validate_runtime_info_wraps_schema_errors_as_runtime_ml_error() -> None:
    """Wrap runtime-info schema failures as ``RuntimeMLError`` with stable messaging."""
    payload = _runtime_info_payload()
    payload["gpu_info"] = {"gpu_count": 1}

    with pytest.raises(RuntimeMLError, match="Invalid runtime info payload"):
        validate_runtime_info(payload)
