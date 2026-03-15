
import os
import subprocess

from fastapi import HTTPException
from ml_service.backend.registries.exit_codes_meaning import EXIT_MEANING
from pydantic import BaseModel

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root

def execute_pipeline(
    module_path: str,
    payload: BaseModel,
    boolean_args: list[str] | None = None,
):
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
            detail=f"Failed to start pipeline: {e}"
        ) from e

    exit_code = result.returncode

    return {
        "exit_code": exit_code,
        "status": EXIT_MEANING.get(exit_code, "UNKNOWN_ERROR"),
        "stdout": result.stdout,
        "stderr": result.stderr,
        }
