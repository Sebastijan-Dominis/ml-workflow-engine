import hashlib

import pandas as pd

def hash_feature_schema(X: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for col in X.columns:
        h.update(col.encode())
        h.update(str(X[col].dtype).encode())
    return h.hexdigest()