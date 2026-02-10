import hashlib
import json

def hash_snapshot_identity(file_hashes: dict[str, str]) -> str:
    payload = json.dumps(file_hashes, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()