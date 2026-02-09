import pynvml

from ml.config.validation_schemas.hardware_cfg import HardwareConfig


def parse_cuda_driver_version(version_int: int) -> str:
    major = version_int // 1000
    minor = (version_int % 1000) // 10
    return f"{major}.{minor}"

def prepare_gpu_info() -> tuple[list[str], list[int], list[int], str, str]:
    gpu_names = []
    gpu_memories_gb = []

    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpu_devices_available = list(range(gpu_count))
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            gpu_names.append(name.decode("utf-8") if isinstance(name, bytes) else name)
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_total_gb = round(int(gpu_mem.total) / 1e9, 2)
            gpu_memories_gb.append(gpu_total_gb)
        cuda_version_int = pynvml.nvmlSystemGetCudaDriverVersion()
        cuda_version_str = parse_cuda_driver_version(cuda_version_int)
        gpu_driver_version = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        gpu_names = []
        gpu_devices_available = []
        gpu_memories_gb = []
        cuda_version_str = "Unknown"
        gpu_driver_version = "Unknown"
    return gpu_names, gpu_devices_available, gpu_memories_gb, cuda_version_str, gpu_driver_version

def get_gpu_info(hardware_info: HardwareConfig) -> dict:
    gpu_info = {}

    gpu_names, gpu_devices_available, gpu_memories_gb, cuda_version_str, gpu_driver_version = prepare_gpu_info()
    gpu_info["task_type"] = hardware_info.task_type.value
    gpu_info["gpu_count"] = len(gpu_names)
    gpu_info["gpu_devices_available"] = gpu_devices_available
    gpu_info["gpu_names"] = gpu_names
    gpu_info["gpu_memories_gb"] = gpu_memories_gb
    gpu_info["gpu_devices_used"] = hardware_info.devices
    gpu_info["cuda_version"] = cuda_version_str
    gpu_info["gpu_driver_version"] = gpu_driver_version

    return gpu_info
