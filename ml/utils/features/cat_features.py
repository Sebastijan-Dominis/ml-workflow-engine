import pandas as pd

def get_cat_features(input_schema: pd.DataFrame, derived_schema: pd.DataFrame) -> list:
    input_categoricals = input_schema.loc[
        input_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    derived_categoricals = derived_schema.loc[
        derived_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    return input_categoricals + derived_categoricals