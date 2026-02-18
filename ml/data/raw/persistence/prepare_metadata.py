from datetime import datetime
from pathlib import Path

import pandas as pd

from ml.data.utils.memory.get_memory_usage import get_memory_usage
from ml.registry.hash_registry import hash_dataset


def prepare_metadata(df: pd.DataFrame, *, args, dataset_path: Path) -> dict:
    dataset_hash = hash_dataset(dataset_path)
    
    timestamp = datetime.now().isoformat()

    metadata = {
        "dataset": {
            "name": args.dataset,
            "version": args.version,
            "path_suffix": args.path_suffix,
            "format": args.format,
            "hash": dataset_hash
        },
        "rows": len(df),
        "columns": {
            "count": len(df.columns),
            "names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict()
        },
        "created_at": timestamp,
        "created_by": "handle_raw.py",
        "owner": args.owner,
        "memory_usage_mb": get_memory_usage(df),
    }

    return metadata