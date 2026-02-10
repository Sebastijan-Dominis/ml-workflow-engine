import hashlib
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from ml.utils.features.hashing.safe import safe

def hash_arrow_metadata(path: Path) -> str:
    with pa.memory_map(path, 'r') as source:
        reader = pa.ipc.open_file(source)
        schema = reader.schema

        h = hashlib.sha256()
        for field in schema:
            h.update(field.name.encode())
            h.update(safe(field.type).encode())
            h.update(safe(field.nullable).encode())

        h.update(safe(reader.num_record_batches).encode())
        return h.hexdigest()