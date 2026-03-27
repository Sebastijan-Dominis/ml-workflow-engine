import pandas as pd
from ml.post_promotion.monitoring.feature_drifting.comparison import compare_feature_distributions


def test_compare_feature_distributions_monkeypatched(monkeypatch):
    # monkeypatch compute_drift in the comparison module
    import ml.post_promotion.monitoring.feature_drifting.comparison as comp_mod

    monkeypatch.setattr(comp_mod, "compute_drift", lambda e, a: 0.42)

    training = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    inference = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})

    out = compare_feature_distributions(stage="production", inference_features=inference, training_features=training)
    assert out["f1"] == 0.42
    assert out["f2"] == 0.42
