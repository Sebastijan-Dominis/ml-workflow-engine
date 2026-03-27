from pathlib import Path

import pandas as pd


def test_inference_and_prediction_dataclasses():
    from ml.post_promotion.inference.classes.function_returns import (
        ArtifactLoadingReturn,
        PredictionStoringReturn,
    )

    alr = ArtifactLoadingReturn(artifact=object(), artifact_hash="h", artifact_type="model")
    assert alr.artifact_hash == "h"

    pr = PredictionStoringReturn(file_path=Path("x"), cols=["a", "b"])
    assert pr.file_path == Path("x")


def test_monitoring_dataclasses():
    from ml.post_promotion.monitoring.classes.function_returns import (
        InferenceFeaturesAndTarget,
        MonitoringExecutionOutput,
    )

    df = pd.DataFrame({"a": [1]})
    s = pd.Series([0])
    ifat = InferenceFeaturesAndTarget(features=df, target=s)
    assert ifat.features.equals(df)

    meo = MonitoringExecutionOutput(drift_results={"a": 0.1}, performance_results={"m": {"v": 1}}, model_version="mv")
    assert meo.model_version == "mv"
