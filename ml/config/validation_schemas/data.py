from typing import Literal

from pydantic import BaseModel


class DataConfig(BaseModel):
    path: str
    metadata_path: str
    source: str
    format: Literal["csv","parquet","json", "arrow"]