"""A module to execute a script as a subprocess and capture its output and exit code."""
import os
import subprocess

from fastapi import HTTPException
from pydantic import BaseModel

from ml_service.backend.registries.exit_codes_meaning import EXIT_MEANING

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root

def execute_script(
    module_path: str,
    payload: BaseModel,
    boolean_args: list[str] | None = None,
):
    """Execute a script as a subprocess and capture its output and exit code.

    Args:
        module_path (str): The module path to execute (e.g., "ml_service.scripts.my_script").
        payload (BaseModel): The payload containing the arguments for the script.
        boolean_args (list[str] | None): A list of argument names that should be treated as boolean flags.

    Returns:
        dict: A dictionary containing the exit code, status, stdout, and stderr of the executed script.
    """
    print(f"Executing script with payload: {payload}")

    boolean_args = boolean_args or []

    cmd = [
        "python", "-m", module_path,
    ]

    for field_name, value in payload.model_dump(exclude_none=True).items():
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue

        flag = f"--{field_name.replace('_', '-')}"

        if field_name in boolean_args:
            value = "True" if value else "False"

        if isinstance(value, list):
            cmd.extend([flag] + [str(v) for v in value])
        else:
            cmd.extend([flag, str(value)])

    print(f"Executing command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_root
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start script: {e}"
        ) from e

    exit_code = result.returncode

    return {
        "exit_code": exit_code,
        "status": EXIT_MEANING.get(exit_code, "UNKNOWN_ERROR"),
        "stdout": result.stdout,
        "stderr": result.stderr,
        }
