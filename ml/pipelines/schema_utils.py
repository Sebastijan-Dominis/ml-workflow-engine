import pandas as pd
from typing import List, Tuple

def get_raw_and_derived_features(input_schema: pd.DataFrame, derived_schema: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (raw_features, derived_features, all_features)
    """
    raw_features = input_schema["feature"].tolist()
    derived_features = derived_schema["feature"].tolist()
    all_features = raw_features + derived_features
    return raw_features, derived_features, all_features

def get_categorical_features(schema: pd.DataFrame) -> List[str]:
    """
    Return list of categorical feature names from a schema DataFrame.
    """
    return schema.loc[schema["dtype"].isin(["object", "string", "category"]), "feature"].tolist()
