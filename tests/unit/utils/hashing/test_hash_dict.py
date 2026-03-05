import pytest
from ml.utils.hashing.hash_dict import hash_dict

pytestmark = pytest.mark.unit


def test_hash_dict_is_order_invariant_for_nested_payloads() -> None:
    payload_a = {
        "thresholds": {
            "val": {"f1": 0.71, "roc_auc": 0.82},
        },
        "sets": ["val", "test"],
        "channels": {"ota", "direct"},
    }
    payload_b = {
        "channels": {"direct", "ota"},
        "sets": ["val", "test"],
        "thresholds": {
            "val": {"roc_auc": 0.82, "f1": 0.71},
        },
    }

    assert hash_dict(payload_a) == hash_dict(payload_b)
