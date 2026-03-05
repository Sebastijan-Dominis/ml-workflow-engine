"""Unit tests for the hash_dict function in ml.utils.hashing.hash_dict, which generates a hash for a given dictionary. The tests verify that the hash is consistent for dictionaries with the same content but different key orders, including nested dictionaries and sets."""
import pytest
from ml.utils.hashing.hash_dict import hash_dict

pytestmark = pytest.mark.unit


def test_hash_dict_is_order_invariant_for_nested_payloads() -> None:
    """Test that hash_dict produces the same hash for two dictionaries with the same content but different key orders, including nested dictionaries and sets."""
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
