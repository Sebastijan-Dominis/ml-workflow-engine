from typing import Optional, Literal

from pydantic import BaseModel, Field

class Input(BaseModel):
    path: str = Field(..., description="Path to the input dataset file.")
    format: str = Field(..., description="Format of the input dataset (e.g., 'csv', 'parquet').")

class Output(BaseModel):
    path_suffix: str = Field(..., description="Suffix for the output dataset file path, which will be combined with the dataset name and version to create the full path.")
    format: Literal["parquet"] = Field("parquet", description="Format to save the interim dataset.")
    compression: Optional[Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd']] = Field(None, description="Compression method to use when saving the dataset (default: 'snappy').")

class DatasetInfo(BaseModel):
    name: str = Field(..., description="Name of the dataset being processed.")
    version: str = Field(..., description="Version of the interim dataset being created.")
    output: Output