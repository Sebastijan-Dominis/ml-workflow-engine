import yaml
from ml_service.backend.configs.promotion_thresholds.utils.check_thresholds_exist import (
    check_thresholds_exist,
)


def test_missing_file(tmp_path):
    p = tmp_path / "no.yaml"
    exists, thresholds = check_thresholds_exist(p, "regression", "s1")
    assert exists is False
    assert thresholds == {}


def test_empty_file(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    exists, thresholds = check_thresholds_exist(p, "regression", "s1")
    assert exists is False
    assert thresholds == {}


def test_missing_segment(tmp_path):
    p = tmp_path / "th.yaml"
    content = {"regression": {"other_segment": {"threshold": 0.1}}}
    p.write_text(yaml.safe_dump(content))
    exists, thresholds = check_thresholds_exist(p, "regression", "s1")
    assert exists is False
    assert "regression" in thresholds
    assert thresholds["regression"] == {"other_segment": {"threshold": 0.1}}


def test_existing_segment(tmp_path):
    p = tmp_path / "th.yaml"
    content = {"regression": {"s1": {"threshold": 0.5}}}
    p.write_text(yaml.safe_dump(content))
    exists, thresholds = check_thresholds_exist(p, "regression", "s1")
    assert exists is True
    assert thresholds["regression"]["s1"] == {"threshold": 0.5}
