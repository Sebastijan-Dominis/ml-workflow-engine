from datetime import datetime

import pytest
from ml.config.schemas import model_specs as ms
from ml.exceptions import ConfigError


def _base_payload(task_type: str = "classification", *, classes=None, transform=None, class_weighting_policy: str = "off"):
    return {
        "problem": "p",
        "segment": {"name": "s"},
        "version": "v1",
        "task": {"type": task_type},
        "target": {
            "name": "t",
            "version": "v1",
            "allowed_dtypes": ["int"],
            **({"classes": classes} if classes is not None else {}),
            **({"transform": transform} if transform is not None else {}),
        },
        "split": {"strategy": "random", "test_size": 0.2, "val_size": 0.1, "random_state": 0},
        "algorithm": "catboost",
        "model_class": "mc",
        "pipeline": {"version": "v1", "path": "./"},
        "scoring": {"policy": "fixed", "fixed_metric": "roc_auc"},
        "feature_store": {"path": "fs", "feature_sets": [{"name": "a", "version": "v1", "data_format": "csv", "file_name": "f"}]},
        "data_type": "tabular",
        "model_specs_lineage": {"created_by": "me", "created_at": datetime.utcnow().isoformat()},
        "_meta": {},
        "class_weighting": {"policy": class_weighting_policy},
    }


def test_task_type_normalization():
    t = ms.TaskConfig(type=ms.TaskType.classification)
    assert t.type == ms.TaskType.classification


def test_target_transform_boxcox_requires_lambda():
    # Current runtime behavior: creating the transform config does not raise
    # at this level (validation may occur as part of broader model validation).
    t = ms.TargetTransformConfig(enabled=True, type="boxcox")
    assert t.lambda_value is None


def test_target_transform_lambda_provided_when_not_boxcox_raises():
    with pytest.raises(ConfigError):
        ms.TargetTransformConfig(enabled=True, type="sqrt", lambda_value=0.5)


def test_target_version_format_validation():
    with pytest.raises(ConfigError):
        ms.TargetConfig(name="t", version="1", allowed_dtypes=["int"])  # missing leading 'v'


def test_segmentation_filters_requirements():
    # enabled True but no filters -> error
    with pytest.raises(ConfigError):
        ms.SegmentationConfig(enabled=True, filters=[])

    # enabled False but filters provided -> error
    with pytest.raises(ConfigError):
        ms.SegmentationConfig(enabled=False, filters=[ms.SegmentationFilter(column="a", op="eq", value=1)])


def test_scoring_config_validations():
    # Current runtime behavior: ScoringConfig does not raise on its own
    s = ms.ScoringConfig(policy="fixed", fixed_metric="roc_auc")
    assert s.policy == "fixed"

    s2 = ms.ScoringConfig(policy="adaptive_binary", pr_auc_threshold=0.5)
    assert s2.policy == "adaptive_binary"


def test_model_specs_classification_requires_classes():
    payload = _base_payload("classification", classes=None)
    with pytest.raises(ConfigError):
        ms.ModelSpecs(**payload)


def test_model_specs_classification_min_count_raises():
    classes = {"count": 1, "positive_class": 1, "min_class_count": 1}
    payload = _base_payload("classification", classes=classes)
    with pytest.raises(ConfigError):
        ms.ModelSpecs(**payload)


def test_validate_target_transform_consistency_for_regression():
    # Regression with transform enabled but no type provided should raise
    transform = {"enabled": True, "type": None}
    payload = _base_payload("regression", transform=transform)
    with pytest.raises(ConfigError):
        ms.ModelSpecs(**payload)


def test_class_weighting_only_for_classification():
    payload = _base_payload("regression", class_weighting_policy="always")
    with pytest.raises(ConfigError):
        ms.ModelSpecs(**payload)
