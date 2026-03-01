import hashlib
import json

def hash_list(lst: list, order_matters: bool = True) -> str:
    if not order_matters:
        lst = sorted(lst)
    normalized = json.dumps(lst, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(normalized.encode()).hexdigest()