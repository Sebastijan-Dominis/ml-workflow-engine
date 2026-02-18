import logging
import hashlib
from pathlib import Path
import pyarrow as pa

from ml.utils.features.hashing.safe import safe
from ml.exceptions import RuntimeMLException

logger = logging.getLogger(__name__)

def hash_arrow_metadata(path: Path) -> str:
    try:
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
    except Exception as e:
        msg = f"Failed to hash Arrow metadata for file {path}. "
        logger.error(msg + f"Details: {str(e)}")
        raise RuntimeMLException(msg) from e