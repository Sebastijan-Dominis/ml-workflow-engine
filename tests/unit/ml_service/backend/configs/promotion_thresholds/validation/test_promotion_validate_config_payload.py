import importlib

import pytest


def _valid_payload():
    return {
        "promotion_metrics": {
            "sets": ["test"],
            "metrics": ["accuracy"],
            "directions": {"accuracy": "maximize"},
        },
        "thresholds": {"test": {"accuracy": 0.9}, "val": {}, "train": {}},
        "lineage": {"created_by": "tester", "created_at": "2024-01-01T00:00:00"},
    }


def test_validate_config_payload_accepts_valid_payload():
    mod = importlib.import_module(
        "ml_service.backend.configs.promotion_thresholds.validation.validate_config_payload"
    )
    res = mod.validate_config_payload(_valid_payload())
    # type check: returns PromotionThresholds model instance
    from ml.promotion.config.promotion_thresholds import PromotionThresholds

    assert isinstance(res, PromotionThresholds)


def test_validate_config_payload_rejects_inconsistent_sets():
    mod = importlib.import_module(
        "ml_service.backend.configs.promotion_thresholds.validation.validate_config_payload"
    )

    payload = _valid_payload()
    # require both test and val to be present but leave val thresholds empty
    payload["promotion_metrics"]["sets"] = ["test", "val"]

    with pytest.raises(Exception) as excinfo:
        mod.validate_config_payload(payload)

    # should raise a ConfigError (subclass of Exception) for inconsistency
    assert excinfo.value is not None
