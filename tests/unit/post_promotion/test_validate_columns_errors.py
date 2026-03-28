import pandas as pd
import pytest
from ml.post_promotion.inference.validation.validate_columns import validate_columns


def test_validate_columns_missing_base_columns():
    df = pd.DataFrame({"run_id": [1]})
    with pytest.raises(ValueError):
        validate_columns(df)


def test_validate_columns_malformed_probability_columns():
    base_cols = {
        "run_id": ["r1"],
        "prediction_id": ["p1"],
        "timestamp": [pd.Timestamp.now()],
        "model_stage": ["production"],
        "model_version": ["v1"],
        "entity_id": ["e1"],
        "input_hash": ["h"],
        "prediction": [0],
        "schema_version": ["1"],
    }

    df = pd.DataFrame(base_cols)
    # malformed: proba_0 and proba_2 -> missing proba_1
    df["proba_0"] = [0.7]
    df["proba_2"] = [0.3]

    with pytest.raises(ValueError):
        validate_columns(df)
