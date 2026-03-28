from datetime import datetime

import pytest
from ml.exceptions import RuntimeMLError
from ml.metadata.schemas.post_promotion.infer import InferenceMetadata
from ml.metadata.validation.post_promotion.infer import validate_inference_metadata


def _example_payload() -> dict:
    return {
        "problem_type": "classification",
        "segment": "s1",
        "model_version": "v1",
        "model_stage": "production",
        "run_id": "r1",
        "timestamp": datetime.now().isoformat(),
        "columns": ["a", "b"],
        "snapshot_bindings_id": "sb",
        "feature_lineage": [
            {
                "name": "f",
                "version": "1",
                "snapshot_id": "s",
                "file_hash": "fh",
                "in_memory_hash": "imh",
                "feature_schema_hash": "fsh",
                "operator_hash": "oh",
                "feature_type": "tabular",
                "file_name": "fn",
                "data_format": "csv",
            }
        ],
        "artifact_type": "model",
        "artifact_hash": "ah",
        "inference_latency_seconds": 0.123,
    }


def test_inference_metadata_model_validate_success():
    payload = _example_payload()
    model = InferenceMetadata.model_validate(payload)
    assert isinstance(model, InferenceMetadata)
    dumped = model.model_dump()
    assert dumped["artifact_type"] == "model"


def test_validate_inference_metadata_raises_on_invalid():
    with pytest.raises(RuntimeMLError):
        validate_inference_metadata({"not": "enough"})
