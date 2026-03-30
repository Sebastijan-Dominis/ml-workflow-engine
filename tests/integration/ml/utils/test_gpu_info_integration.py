"""Integration tests for GPU helpers in `ml.utils.runtime.gpu_info`."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

from ml.config.schemas.hardware_cfg import HardwareConfig, HardwareTaskType


def test_parse_cuda_driver_version_examples() -> None:
    from ml.utils.runtime.gpu_info import parse_cuda_driver_version

    assert parse_cuda_driver_version(11040) == "11.4"
    assert parse_cuda_driver_version(10000) == "10.0"
    assert parse_cuda_driver_version(11000) == "11.0"
    assert parse_cuda_driver_version(11010) == "11.1"


def test_prepare_gpu_info_with_fake_pynvml(monkeypatch: Any) -> None:
    """Simulate `pynvml` functions to exercise `prepare_gpu_info` without hardware."""

    fake = types.SimpleNamespace()

    class NVMLError(Exception):
        pass

    fake.NVMLError = NVMLError
    fake.nvmlInit = lambda: None
    fake.nvmlDeviceGetCount = lambda: 2
    fake.nvmlDeviceGetHandleByIndex = lambda i: i
    fake.nvmlDeviceGetName = lambda h: b"FakeGPU0" if h == 0 else "FakeGPU1"

    def fake_mem(h):
        return types.SimpleNamespace(total=8_000_000_000 if h == 0 else 16_000_000_000)

    fake.nvmlDeviceGetMemoryInfo = fake_mem
    fake.nvmlSystemGetCudaDriverVersion = lambda: 11040
    fake.nvmlSystemGetDriverVersion = lambda: b"470.57.02"
    fake.nvmlShutdown = lambda: None

    monkeypatch.setitem(sys.modules, "pynvml", fake)

    # reload module so it picks up our injected fake module
    import ml.utils.runtime.gpu_info as gpu_info

    importlib.reload(gpu_info)

    names, devices, memories, cuda_str, drv = gpu_info.prepare_gpu_info()

    assert names == ["FakeGPU0", "FakeGPU1"]
    assert devices == [0, 1]
    assert memories == [round(8_000_000_000 / 1e9, 2), round(16_000_000_000 / 1e9, 2)]
    assert cuda_str == "11.4"
    assert isinstance(drv, str)


def test_get_gpu_info_assembles_payload(monkeypatch: Any) -> None:
    import ml.utils.runtime.gpu_info as gpu_info

    # Provide deterministic prepare_gpu_info output
    monkeypatch.setattr(
        gpu_info, "prepare_gpu_info", lambda: (["G0"], [0], [8.0], "11.4", "470.57.02")
    )

    hw = HardwareConfig(task_type=HardwareTaskType.GPU, devices=[0])
    payload = gpu_info.get_gpu_info(hw)

    assert payload["task_type"] == "GPU"
    assert payload["gpu_count"] == 1
    assert payload["gpu_devices_available"] == [0]
    assert payload["gpu_devices_used"] == [0]
    assert payload["cuda_version"] == "11.4"
