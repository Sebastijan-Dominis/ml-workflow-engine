import os
import subprocess
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException
from ml_service.backend.registries.pipelines_for_endpoint_registration import (
    PIPELINES_FOR_ENDPOINT_REGISTRATION,
)
from pydantic import BaseModel

router = APIRouter(prefix="/pipelines", tags=["pipelines"])

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root

ml_environment_name = os.getenv("ML_ENVIRONMENT_NAME", "hotel_management")

EXIT_MEANING = {
    0: "SUCCESS",
    1: "UNEXPECTED_ERROR",
    2: "CONFIG_ERROR",
    3: "DATA_ERROR",
    4: "PIPELINE_ERROR",
    5: "SEARCH_ERROR",
    6: "TRAINING_ERROR",
    7: "EVALUATION_ERROR",
    8: "EXPLAINABILITY_ERROR",
    9: "PERSISTENCE_ERROR",
}

def register_pipeline(
    name: str,
    module_path: str,
    args_schema: type[BaseModel],
    boolean_args: list[str] | None = None,
):
    boolean_args = boolean_args or []

    async def endpoint(payload: Annotated[args_schema, Body(...)]):  # type: ignore
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
                cwd=repo_root,
                shell=True,
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

    router.post(f"/{name}")(endpoint)


for pipeline_info in PIPELINES_FOR_ENDPOINT_REGISTRATION:
    register_pipeline(
        name=pipeline_info["name"],
        module_path=pipeline_info["module_path"],
        args_schema=pipeline_info["args_schema"],
        boolean_args=pipeline_info.get("boolean_args", []),
    )
