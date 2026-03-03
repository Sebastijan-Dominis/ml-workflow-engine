"""Registry mapping file formats to pandas reader callables."""

import pandas as pd

FORMAT_REGISTRY_READ = {
    "parquet": pd.read_parquet,
    "csv": pd.read_csv,
    "json": pd.read_json,
    "arrow": lambda p: pd.read_feather(p),
}