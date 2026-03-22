"""A module containing the input models for the scripts CLI."""
from pydantic import BaseModel


class GenerateColsForRowIdFingerprintInput(BaseModel):
    """Model for the input of the generate_cols_for_row_id_fingerprint script."""
    pass

class GenerateFakeDataInput(BaseModel):
    """Model for the input of the generate_fake_data script."""
    data: str
    version: str
    snapshot_id: str | None = None
    num_rows: int | None = None
    include_old: bool | None = False
    model_path: str | None = None
    strict_missing: bool | None = False
    strict_quality: bool | None = True
    seed: int | None = None
    quality_threshold: float | None = None
    epochs: int | None = None
    batch_target_size: int | None = None
    save_model: bool | None = True

class GenerateOperatorHashInput(BaseModel):
    """Model for the input of the generate_operator_hash script."""
    operators: list[str]

class GenerateSnapshotBindingInput(BaseModel):
    """Model for the input of the generate_snapshot_binding script."""
    pass

class CheckImportLayersInput(BaseModel):
    """Model for the input of the check_import_layers script."""
    pass

class CheckNamingConventionsInput(BaseModel):
    """Model for the input of the check_naming_conventions script."""
    pass
