import hashlib
import json

def canonicalize(obj):
    if isinstance(obj, dict):
        return {k: canonicalize(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, list):
        return [canonicalize(x) for x in obj]
    elif isinstance(obj, set):
        return sorted(canonicalize(x) for x in obj)
    else:
        return obj

def hash_dict(d: dict) -> str:
    normalized = json.dumps(
        canonicalize(d),
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode()).hexdigest()