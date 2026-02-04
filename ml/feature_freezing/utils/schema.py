import hashlib
import pandas as pd

def hash_data_schema(X: pd.DataFrame) -> str:
    arr = pd.util.hash_pandas_object(X, index=True).to_numpy()
    return hashlib.md5(arr.tobytes()).hexdigest()