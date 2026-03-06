"""Unit tests for runtime and metrics model schema contracts."""

from __future__ import annotations

from typing import cast

import pytest
from ml.modeling.models.metrics import EvaluationMetrics, TrainingMetrics
from ml.modeling.models.runtime_info import RuntimeInfo
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _runtime_info_payload() -> dict[str, object]:
    """Build a valid raw runtime-info payload consumed by RuntimeInfo."""
    return {
        "environment": {
            "conda_env_export": "name: hotel_management",
            "conda_env_hash": "env-hash-123",
        },
        "execution": {
            "created_at": "2026-03-06T12:34:56Z",
            "duration_seconds": 42.5,
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


def test_runtime_info_constructs_nested_models_from_raw_dict_payload() -> None:
    """Coerce nested dictionaries into typed runtime submodels with stable field values."""
    runtime_info = RuntimeInfo.model_validate(_runtime_info_payload())

    assert runtime_info.execution.git_commit == "abc123"
    assert runtime_info.gpu_info.task_type.value == "GPU"
    assert runtime_info.runtime.python_build == ("main", "Feb 1 2026 00:00:00")


def test_runtime_info_rejects_invalid_task_type_literal() -> None:
    """Reject unsupported GPU task type values to enforce hardware enum contract."""
    payload = _runtime_info_payload()
    gpu_info = cast(dict[str, object], payload["gpu_info"])
    payload["gpu_info"] = {
        **gpu_info,
        "task_type": "gpu",
    }

    with pytest.raises(ValidationError, match="task_type"):
        RuntimeInfo.model_validate(payload)


def test_training_metrics_accepts_flat_metric_mapping() -> None:
    """Accept flat metric dictionaries for common aggregate training metrics."""
    training_metrics = TrainingMetrics.model_validate(
        {
            "task_type": "classification",
            "algorithm": "catboost",
            "metrics": {"accuracy": 0.89, "f1": 0.84},
        }
    )

    assert training_metrics.metrics["accuracy"] == pytest.approx(0.89)


def test_training_metrics_accepts_nested_per_group_metric_mapping() -> None:
    """Accept nested metric dictionaries for per-group/per-class reporting structures."""
    training_metrics = TrainingMetrics.model_validate(
        {
            "task_type": "classification",
            "algorithm": "catboost",
            "metrics": {
                "class_0": {"precision": 0.91, "recall": 0.88},
                "class_1": {"precision": 0.87, "recall": 0.90},
            },
        }
    )

    class_0 = training_metrics.metrics["class_0"]
    assert isinstance(class_0, dict)
    assert class_0["precision"] == pytest.approx(0.91)


def test_training_metrics_rejects_mixed_flat_and_nested_metric_shapes() -> None:
    """Reject mixed metric-shape payloads that violate union branch consistency."""
    payload = {
        "task_type": "classification",
        "algorithm": "catboost",
        "metrics": {
            "accuracy": 0.89,
            "per_class": {"class_0": 0.90, "class_1": 0.88},
        },
    }

    with pytest.raises(ValidationError, match="metrics"):
        TrainingMetrics.model_validate(payload)


def test_evaluation_metrics_requires_all_standard_splits() -> None:
    """Require train/val/test split metrics for consistent downstream persistence contracts."""
    with pytest.raises(ValidationError, match="test"):
        EvaluationMetrics.model_validate(
            {
                "task_type": "classification",
                "algorithm": "catboost",
                "train": {"accuracy": 0.91},
                "val": {"accuracy": 0.88},
            }
        )
