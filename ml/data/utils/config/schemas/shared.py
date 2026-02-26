from typing import Optional, Literal

from pydantic import BaseModel, Field

class Output(BaseModel):
    path_suffix: str = Field(..., description="Suffix for the output data file path, which will be combined with the data name and version to create the full path.")
    format: Literal["parquet"] = Field("parquet", description="Format to save the interim data.")
    compression: Optional[Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd']] = Field(None, description="Compression method to use when saving the data (default: 'snappy').")

class DataInfo(BaseModel):
    name: str = Field(..., description="Name of the data being processed.")
    version: str = Field(..., description="Version of the interim data being created.")
    output: Output