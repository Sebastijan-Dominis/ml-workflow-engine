import hashlib

import pandas as pd

def hash_dataframe_content(X: pd.DataFrame) -> str:
    arr = pd.util.hash_pandas_object(X, index=False).to_numpy()
    return hashlib.md5(arr.tobytes()).hexdigest()
