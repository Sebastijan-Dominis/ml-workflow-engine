"""Unit tests for GPU runtime information helpers."""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
from ml.config.schemas.hardware_cfg import HardwareConfig

# Ensure gpu_info module can be imported even when pynvml is not installed.
if "pynvml" not in sys.modules:
    pynvml_stub = types.ModuleType("pynvml")
    pynvml_stub.__dict__.update(
        {
            "NVMLError": Exception,
            "nvmlInit": lambda: None,
            "nvmlDeviceGetCount": lambda: 0,
            "nvmlShutdown": lambda: None,
            "nvmlDeviceGetHandleByIndex": lambda index: index,
            "nvmlDeviceGetName": lambda handle: b"GPU",
            "nvmlDeviceGetMemoryInfo": lambda handle: SimpleNamespace(total=8_000_000_000),
            "nvmlSystemGetCudaDriverVersion": lambda: 12040,
            "nvmlSystemGetDriverVersion": lambda: b"550.54",
        }
    )
    sys.modules["pynvml"] = pynvml_stub


gpu_info_module = importlib.import_module("ml.utils.runtime.gpu_info")
get_gpu_info = gpu_info_module.get_gpu_info
parse_cuda_driver_version = gpu_info_module.parse_cuda_driver_version
prepare_gpu_info = gpu_info_module.prepare_gpu_info


pytestmark = pytest.mark.unit


def _cpu_hardware() -> HardwareConfig:
    """Build a CPU-only `HardwareConfig` for tests."""
    return HardwareConfig.model_validate({"task_type": "CPU", "devices": []})


def test_parse_cuda_driver_version_formats_major_minor() -> None:
    """Verify CUDA driver version formatting from integer to `major.minor`."""
    assert parse_cuda_driver_version(12040) == "12.4"


def test_prepare_gpu_info_collects_expected_gpu_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `prepare_gpu_info` returns expected GPU metadata from NVML."""
    monkeypatch.setattr("ml.utils.runtime.gpu_info.pynvml.nvmlInit", lambda: None)
    monkeypatch.setattr("ml.utils.runtime.gpu_info.pynvml.nvmlDeviceGetCount", lambda: 2)
    monkeypatch.setattr("ml.utils.runtime.gpu_info.pynvml.nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(
        "ml.utils.runtime.gpu_info.pynvml.nvmlDeviceGetName",
        lambda handle: b"GPU-A" if handle == 0 else b"GPU-B",
    )
    monkeypatch.setattr(
        "ml.utils.runtime.gpu_info.pynvml.nvmlDeviceGetMemoryInfo",
        lambda handle: SimpleNamespace(total=16_000_000_000 if handle == 0 else 8_000_000_000),
    )
    monkeypatch.setattr("ml.utils.runtime.gpu_info.pynvml.nvmlSystemGetCudaDriverVersion", lambda: 12040)
    monkeypatch.setattr("ml.utils.runtime.gpu_info.pynvml.nvmlSystemGetDriverVersion", lambda: b"550.54")
    monkeypatch.setattr("ml.utils.runtime.gpu_info.pynvml.nvmlShutdown", lambda: None)

    names, devices, memories, cuda_version, driver_version = prepare_gpu_info()

    assert names == ["GPU-A", "GPU-B"]
    assert devices == [0, 1]
    assert memories == [16.0, 8.0]
    assert cuda_version == "12.4"
    assert driver_version == "550.54"


def test_prepare_gpu_info_returns_unknown_fields_on_nvml_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fallback values when NVML raises an error."""
    class _NvmlError(Exception):
        pass

    monkeypatch.setattr("ml.utils.runtime.gpu_info.pynvml.NVMLError", _NvmlError)

    def _raise() -> None:
        """Simulate an NVML failure."""
        raise _NvmlError("nvml failed")

    monkeypatch.setattr("ml.utils.runtime.gpu_info.pynvml.nvmlInit", _raise)

    names, devices, memories, cuda_version, driver_version = prepare_gpu_info()

    assert names == []
    assert devices == []
    assert memories == []
    assert cuda_version == "Unknown"
    assert driver_version == "Unknown"


def test_get_gpu_info_merges_hardware_usage_with_runtime_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that `get_gpu_info` combines hardware selection with runtime metadata."""
    monkeypatch.setattr(
        "ml.utils.runtime.gpu_info.prepare_gpu_info",
        lambda: (["GPU-A"], [0], [16.0], "12.4", "550.54"),
    )
    hardware = HardwareConfig.model_validate({"task_type": "GPU", "devices": [0]})

    info = get_gpu_info(hardware)

    assert info == {
        "task_type": "GPU",
        "gpu_count": 1,
        "gpu_devices_available": [0],
        "gpu_names": ["GPU-A"],
        "gpu_memories_gb": [16.0],
        "gpu_devices_used": [0],
        "cuda_version": "12.4",
        "gpu_driver_version": "550.54",
    }


def test_get_gpu_info_reflects_cpu_mode_with_no_requested_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that CPU mode reports no GPU devices in use."""
    monkeypatch.setattr(
        "ml.utils.runtime.gpu_info.prepare_gpu_info",
        lambda: ([], [], [], "Unknown", "Unknown"),
    )

    info = get_gpu_info(_cpu_hardware())

    assert info["task_type"] == "CPU"
    assert info["gpu_devices_used"] == []
