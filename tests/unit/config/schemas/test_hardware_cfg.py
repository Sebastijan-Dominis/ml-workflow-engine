"""Unit tests for hardware execution configuration schema."""

import pytest
from ml.config.schemas.hardware_cfg import HardwareConfig
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def test_hardware_config_normalizes_cpu_task_type_to_uppercase() -> None:
    """Test that the HardwareConfig schema normalizes the CPU task type to uppercase."""
    cfg = HardwareConfig.model_validate({"task_type": "cpu"})

    assert cfg.task_type == "CPU"


def test_hardware_config_normalizes_gpu_task_type_to_uppercase() -> None:
    """Test that the HardwareConfig schema normalizes the GPU task type to uppercase."""
    cfg = HardwareConfig.model_validate({"task_type": "gPu", "devices": [0, 1]})

    assert cfg.task_type == "GPU"
    assert cfg.devices == [0, 1]


def test_hardware_config_uses_expected_defaults() -> None:
    """Test that the HardwareConfig schema uses expected default values for devices, memory limit, and allow_growth."""
    cfg = HardwareConfig.model_validate({"task_type": "GPU"})

    assert cfg.devices == []
    assert cfg.memory_limit_gb is None
    assert cfg.allow_growth is False


def test_hardware_config_rejects_unknown_task_type() -> None:
    """Test that the HardwareConfig schema raises a ValidationError when an unknown task type is provided."""
    with pytest.raises(ValidationError):
        HardwareConfig.model_validate({"task_type": "TPU"})
