from pathlib import Path

from ml_service.backend.configs.data.utils.get_config_path import get_config_path


def test_get_config_path_basic():
    repo_root = "repo_root"
    config_type = "interim"
    dataset_name = "dataset"
    dataset_version = "v1"

    p = get_config_path(
        repo_root=repo_root,
        config_type=config_type,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )

    expected = (
        Path(repo_root) / "configs" / "data" / config_type / dataset_name / f"{dataset_version}.yaml"
    )

    assert p == expected


def test_get_config_path_trailing_separator(tmp_path):
    repo_root = str(tmp_path) + "/"
    p = get_config_path(
        repo_root=repo_root, config_type="processed", dataset_name="d", dataset_version="v2"
    )

    expected = Path(str(tmp_path)) / "configs" / "data" / "processed" / "d" / "v2.yaml"
    assert p == expected
