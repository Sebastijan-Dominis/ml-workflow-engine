"""Shared schema blocks reused by interim and processed data configs."""

from typing import Literal

from pydantic import BaseModel, Field


class Output(BaseModel):
    """Output artifact settings for persisted datasets."""

    path_suffix: str = Field(..., description="Suffix for the output data file path, which will be combined with the data name and version to create the full path.")
    format: Literal["parquet"] = Field("parquet", description="Format to save the interim data.")
    compression: Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd'] | None = Field(None, description="Compression method to use when saving the data (default: 'snappy').")

class DataInfo(BaseModel):
    """Dataset identity and output target metadata."""

    name: str = Field(..., description="Name of the data being processed.")
    version: str = Field(..., description="Version of the interim data being created.")
    output: Output
