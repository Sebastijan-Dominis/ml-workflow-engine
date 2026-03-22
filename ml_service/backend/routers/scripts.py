"""A module containing all script-related API endpoints."""
from typing import Annotated

from fastapi import APIRouter, Body, Request

from ml_service.backend.main import limiter
from ml_service.backend.scripts.execute_script import execute_script
from ml_service.backend.scripts.models.script_cli_args import (
    CheckImportLayersInput,
    CheckNamingConventionsInput,
    GenerateColsForRowIdFingerprintInput,
    GenerateFakeDataInput,
    GenerateOperatorHashInput,
    GenerateSnapshotBindingInput,
)

router = APIRouter(prefix="/scripts", tags=["scripts"])

@router.post("/generate_cols_for_row_id_fingerprint", status_code=200)
@limiter.limit("1/15seconds")
def generate_cols_for_row_id_fingerprint(payload: Annotated[GenerateColsForRowIdFingerprintInput, Body(...)], request: Request): # type: ignore
    """Generates the columns for the row ID fingerprint by executing the corresponding script."""
    return execute_script(
        module_path="scripts.generators.generate_cols_for_row_id_fingerprint",
        payload=payload,
        boolean_args=[],
    )

@router.post("/generate_fake_data", status_code=200)
@limiter.limit("1/5minutes")
def generate_fake_data(payload: Annotated[GenerateFakeDataInput, Body(...)], request: Request): # type: ignore
    """Generates fake data by executing the corresponding script."""
    return execute_script(
        module_path="scripts.generators.generate_fake_data",
        payload=payload,
        boolean_args=["include_old", "strict_missing", "strict_quality", "save_model"],
    )

@router.post("/generate_operator_hash", status_code=200)
@limiter.limit("1/5seconds")
def generate_operator_hash(payload: Annotated[GenerateOperatorHashInput, Body(...)], request: Request): # type: ignore
    """Generates the operator hash by executing the corresponding script."""
    return execute_script(
        module_path="scripts.generators.generate_operator_hash",
        payload=payload,
        boolean_args=[],
    )

@router.post("/generate_snapshot_binding", status_code=200)
@limiter.limit("1/2minutes")
def generate_snapshot_binding(payload: Annotated[GenerateSnapshotBindingInput, Body(...)], request: Request): # type: ignore
    """Generates the snapshot binding by executing the corresponding script."""
    return execute_script(
        module_path="scripts.generators.generate_snapshot_binding",
        payload=payload,
        boolean_args=[],
    )

@router.post("/check_import_layers", status_code=200)
@limiter.limit("1/5seconds")
def check_import_layers(payload: Annotated[CheckImportLayersInput, Body(...)], request: Request): # type: ignore
    """Checks the import layers by executing the corresponding script."""
    return execute_script(
        module_path="scripts.quality.check_import_layers",
        payload=payload,
        boolean_args=[],
    )

@router.post("/check_naming_conventions", status_code=200)
@limiter.limit("1/5seconds")
def check_naming_conventions(payload: Annotated[CheckNamingConventionsInput, Body(...)], request: Request): # type: ignore
    """Checks the naming conventions by executing the corresponding script."""
    return execute_script(
        module_path="scripts.quality.check_naming_conventions",
        payload=payload,
        boolean_args=[],
    )
