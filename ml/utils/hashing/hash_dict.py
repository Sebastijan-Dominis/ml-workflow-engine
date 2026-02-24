import hashlib

def hash_dict(d: dict) -> str:
    """Compute a hash for a dictionary."""
    return hashlib.sha256(str(d).encode()).hexdigest()