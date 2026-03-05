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
    """Helper function to create a HardwareConfig instance representing CPU hardware with no GPU devices.

    Returns:
        HardwareConfig: A HardwareConfig instance with task_type set to "CPU" and an empty list of devices, representing a CPU hardware configuration with no GPU devices requested.
    """
    return HardwareConfig.model_validate({"task_type": "CPU", "devices": []})


def test_parse_cuda_driver_version_formats_major_minor() -> None:
    """Test that the parse_cuda_driver_version function correctly formats a CUDA driver version integer into a major.minor string format. The test provides a sample CUDA driver version integer (e.g., 12040) and asserts that the output string is correctly formatted as "12.4", indicating that the function correctly extracts the major and minor version numbers from the integer representation."""
    assert parse_cuda_driver_version(12040) == "12.4"


def test_prepare_gpu_info_collects_expected_gpu_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that prepare_gpu_info successfully collects GPU metadata using the pynvml library and returns the expected information. The test uses monkeypatch to replace the relevant pynvml functions with fake implementations that return controlled outputs, simulating a system with two GPUs (GPU-A and GPU-B) with specific memory sizes, CUDA driver version, and driver version. It then calls prepare_gpu_info and asserts that the returned GPU names, device indices, memory sizes in GB, CUDA version string, and driver version string match the expected values based on the fake pynvml implementations. This validates that prepare_gpu_info correctly interacts with pynvml to gather GPU information and formats it as expected."""
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
    """Test that if any of the pynvml functions called within prepare_gpu_info raise an NVMLError, prepare_gpu_info catches this and returns empty lists for GPU names, devices, and memories, and "Unknown" for CUDA version and driver version. The test uses monkeypatch to replace one of the pynvml functions (e.g., nvmlInit) with a fake function that raises a custom exception simulating an NVMLError, then calls prepare_gpu_info and asserts that the returned values match the expected fallback values (empty lists and "Unknown" strings). This validates that prepare_gpu_info correctly handles exceptions from pynvml and returns appropriate fallback values when GPU information cannot be collected."""
    class _NvmlError(Exception):
        pass

    monkeypatch.setattr("ml.utils.runtime.gpu_info.pynvml.NVMLError", _NvmlError)

    def _raise() -> None:
        """Fake function to simulate an NVMLError being raised by a pynvml function."""
        raise _NvmlError("nvml failed")

    monkeypatch.setattr("ml.utils.runtime.gpu_info.pynvml.nvmlInit", _raise)

    names, devices, memories, cuda_version, driver_version = prepare_gpu_info()

    assert names == []
    assert devices == []
    assert memories == []
    assert cuda_version == "Unknown"
    assert driver_version == "Unknown"


def test_get_gpu_info_merges_hardware_usage_with_runtime_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_gpu_info correctly merges the GPU usage information from the provided HardwareConfig with the GPU metadata collected by prepare_gpu_info. The test uses monkeypatch to replace prepare_gpu_info with a fake function that returns specific GPU metadata (e.g., one GPU named "GPU-A" with 16 GB of memory, CUDA version "12.4", and driver version "550.54"), then creates a HardwareConfig instance representing a GPU task with one device (device index 0). It calls get_gpu_info with this hardware configuration and asserts that the returned dictionary contains the expected merged information, including the task type, GPU count, available devices, GPU names, memory sizes, devices used based on the hardware config, CUDA version, and driver version. This validates that get_gpu_info correctly integrates the hardware usage information with the runtime GPU metadata to produce a comprehensive summary of the GPU information for the current runtime environment."""
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
    """Test that get_gpu_info correctly reflects CPU mode when the provided HardwareConfig has task_type "CPU" and no GPU devices requested. The test uses monkeypatch to replace prepare_gpu_info with a fake function that returns empty GPU metadata, then creates a HardwareConfig instance representing CPU hardware with no devices, calls get_gpu_info with this hardware configuration, and asserts that the returned dictionary indicates a task type of "CPU" and an empty list for gpu_devices_used. This validates that get_gpu_info correctly identifies CPU mode and does not report any GPU devices used when the hardware configuration specifies CPU usage."""
    monkeypatch.setattr(
        "ml.utils.runtime.gpu_info.prepare_gpu_info",
        lambda: ([], [], [], "Unknown", "Unknown"),
    )

    info = get_gpu_info(_cpu_hardware())

    assert info["task_type"] == "CPU"
    assert info["gpu_devices_used"] == []
