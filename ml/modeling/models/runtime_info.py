"""Runtime information about the system where the runner is executing."""
from ml.config.schemas.hardware_cfg import HardwareTaskType
from pydantic import BaseModel


class Environment(BaseModel):
    """Environment information about the conda environment where the runner is executing."""
    conda_env_export: str
    conda_env_hash: str

class Execution(BaseModel):
    """Execution information about the run being executed."""
    created_at: str
    duration_seconds: float
    git_commit: str
    python_executable: str

class GpuInfo(BaseModel):
    """GPU information about the system where the runner is executing."""
    cuda_version: str
    gpu_count: int
    gpu_devices_available: list[int]
    gpu_devices_used: list[int]
    gpu_driver_version: str
    gpu_memories_gb: list[float]
    gpu_names: list[str]
    task_type: HardwareTaskType

class Runtime(BaseModel):
    """Runtime information about the system where the runner is executing."""
    os: str
    os_release: str
    architecture: str
    processor: str
    ram_total_gb: float
    platform_string: str
    hostname: str
    python_version: str
    python_impl: str
    python_build: tuple[str, str]

class RuntimeInfo(BaseModel):
    """Comprehensive runtime information about the system where the runner is executing."""
    environment: Environment
    execution: Execution
    gpu_info: GpuInfo
    runtime: Runtime
