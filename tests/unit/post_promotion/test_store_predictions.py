from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow.parquet as pq
import pytest
from ml.post_promotion.inference.classes.function_returns import PredictionStoringReturn
from ml.post_promotion.inference.persistence.store_predictions import (
    store_predictions,
)
from ml.post_promotion.inference.validation.validate_columns import validate_columns
from ml.promotion.config.registry_entry import RegistryEntry


def make_simple_preds():
    return pd.DataFrame(
        {
            "entity_key": ["e1", "e2"],
            "prediction": [0, 1],
            "probability_0": [0.8, 0.2],
            "probability_1": [0.2, 0.8],
        }
    )


def test_validate_columns_accepts_expected_prob_cols():
    df = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "timestamp": [pd.Timestamp.now(), pd.Timestamp.now()],
            "prediction_id": ["p1", "p2"],
            "model_stage": ["production", "production"],
            "model_version": ["v1", "v1"],
            "entity_id": ["e1", "e2"],
            "input_hash": ["h1", "h2"],
            "prediction": [0, 1],
            "proba_0": [0.8, 0.2],
            "proba_1": [0.2, 0.8],
            "schema_version": ["1", "1"],
        }
    )
    # should not raise and should return a list of columns
    cols = validate_columns(df)
    assert isinstance(cols, list)


def test_store_predictions_builds_return_object(tmp_path, monkeypatch):
    df = make_simple_preds()
    out_dir = tmp_path / "out"

    def fake_write_table(table, path, **kwargs):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ok")

    monkeypatch.setattr(pq, "write_table", fake_write_table)

    ret = store_predictions(
        features=df,
        entity_key="entity_key",
        run_id="r1",
        input_hash=pd.Series(["h1", "h2"]),
        path=out_dir,
        timestamp=pd.Timestamp.now(),
        predictions=df["prediction"],
        probabilities=df[["probability_0", "probability_1"]],
        model_metadata=cast(
            RegistryEntry, type("M", (), {"model_version": "v1"})()
        ),
        stage="production",
    )
    assert isinstance(ret, PredictionStoringReturn)
    assert ret.file_path.exists()


def test_validate_columns_rejects_missing_entity_key():
    df = pd.DataFrame({"prediction": [1]})
    with pytest.raises(ValueError):
        validate_columns(df)
