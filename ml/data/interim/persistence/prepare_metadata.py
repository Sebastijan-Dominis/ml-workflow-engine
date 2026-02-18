import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.data.utils.config.schemas.interim import InterimConfig
from ml.registry.hash_registry import hash_dataset
from ml.utils.compute_config_hash import compute_config_hash

logger = logging.getLogger(__name__)
def prepare_metadata(df: pd.DataFrame, *, config: InterimConfig, start_time: float, dataset_path: Path, owner: str, memory_info: dict) -> dict:
    dataset_hash = hash_dataset(dataset_path)
         
    config_hash = compute_config_hash(config)
    
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")

    duration = time.perf_counter() - start_time

    metadata = {
        "source_dataset": {
            "path": config.input.path,
            "format": config.input.format,
        },
        "dataset": {
            "name": config.dataset.name,
            "version": config.dataset.version,
            "output": {
                "path_suffix": config.dataset.output.path_suffix,
                "format": config.dataset.output.format,
                "compression": config.dataset.output.compression,
            },
            "hash": dataset_hash,
        },
        "memory": memory_info,
        "rows": len(df),
        "columns": {
            "count": len(df.columns),
            "names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict()
        },
        "config_hash": config_hash,
        "created_at": timestamp,
        "created_by": "make_interim.py",
        "owner": owner,
        "duration": duration,
        "runtime_info": {
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "yaml_version": yaml.__version__,
            "python_version": platform.python_version(),
        }
    }

    return metadata