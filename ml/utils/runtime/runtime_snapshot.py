import logging
logger = logging.getLogger(__name__)
import subprocess
import hashlib
import sys
import shutil
from pathlib import Path
import os
from datetime import datetime
import platform

from ml.exceptions import RuntimeMLException
from ml.utils.runtime.runtime_info import get_runtime_info
from ml.utils.git import get_git_commit


def find_conda_executable():
    conda = shutil.which("conda")
    if conda:
        return conda

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        base = Path(conda_prefix).parent.parent
        bin_dir = "bin" if platform.system() != "Windows" else "Scripts"
        candidate = base / bin_dir / "conda"
        if candidate.exists():
            return str(candidate)

    raise RuntimeMLException("Could not locate conda executable.")

def _run_command(cmd: list[str]) -> str:
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except Exception as e:
        msg = f"Command failed: {' '.join(cmd)} | {e}\nstdout: {getattr(e, 'stdout', '')}\nstderr: {getattr(e, 'stderr', '')}"
        logger.error(msg)
        raise RuntimeMLException(msg)


def get_conda_env_export() -> str:
    """
    Returns raw YAML string from:
        conda env export
    """
    try:
        conda = find_conda_executable()
        return _run_command([conda, "env", "export", "--no-builds"])
    except Exception as e:
        msg = f"Failed to export conda environment: {e}"
        logger.error(msg)
        raise RuntimeMLException(msg)


def hash_environment(env_export: str) -> str:
    return hashlib.sha256(env_export.encode()).hexdigest()


def build_runtime_snapshot(timestamp: str) -> dict:
    try:
        runtime = {
            "execution": {
                "created_at": timestamp,
                "git_commit": get_git_commit(Path(".")),
                "python_executable": sys.executable
            },
            "runtime": get_runtime_info(),
        }

        try:
            env_export = get_conda_env_export()
            runtime["environment"] = {
                "conda_env_export": env_export,
                "conda_env_hash": hash_environment(env_export),
            }
        except Exception as e:
            logger.warning(f"Skipping conda_env_export: {e}")

        return runtime
    except Exception as e:
        msg = f"Failed to build runtime snapshot: {e}"
        logger.error(msg)
        raise RuntimeMLException(msg)