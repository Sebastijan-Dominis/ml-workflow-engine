from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow.parquet as pq
from ml.post_promotion.inference.persistence.store_predictions import store_predictions
from ml.promotion.config.registry_entry import RegistryEntry


def test_store_predictions_writes_parquet_and_returns_cols(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": ["e1", "e2"], "f1": [0.1, 0.2]})
    input_hash = pd.Series(["h1", "h2"])
    preds = pd.Series([0, 1])
    probs = pd.DataFrame([[0.1, 0.9], [0.8, 0.2]])
    probs.columns = ["p0", "p1"]

    out_dir = tmp_path / "out"
    timestamp = datetime.utcnow()

    model_metadata = cast(RegistryEntry, type("M", (), {"model_version": "v1"})())

    ret = store_predictions(
        features=df,
        entity_key="id",
        run_id="r1",
        input_hash=input_hash,
        path=out_dir,
        timestamp=timestamp,
        predictions=preds,
        probabilities=probs,
        model_metadata=model_metadata,
        stage="production",
    )

    assert ret.file_path.exists()

    table = pq.read_table(ret.file_path)
    df_read = table.to_pandas()

    assert "run_id" in df_read.columns
    assert "entity_id" in df_read.columns
    assert "prediction" in df_read.columns
