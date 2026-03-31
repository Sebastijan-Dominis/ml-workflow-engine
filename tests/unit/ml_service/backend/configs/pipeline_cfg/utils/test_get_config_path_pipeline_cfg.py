from pathlib import Path

from ml_service.backend.configs.pipeline_cfg.utils.get_config_path import (
    get_config_path,
)


def test_get_config_path_basic():
    repo_root = "repo_root"
    data_type = "tabular"
    algorithm = "random_forest"
    pipeline_version = "v1.0"

    p = get_config_path(
        repo_root=repo_root,
        data_type=data_type,
        algorithm=algorithm,
        pipeline_version=pipeline_version,
    )

    expected = (
        Path(repo_root)
        / "configs"
        / "pipelines"
        / data_type
        / algorithm
        / f"{pipeline_version}.yaml"
    )

    assert isinstance(p, Path)
    assert p == expected


def test_get_config_path_trailing_separator(tmp_path):
    repo_root = str(tmp_path) + "/"
    p = get_config_path(
        repo_root=repo_root, data_type="dt", algorithm="alg", pipeline_version="v2"
    )

    expected = Path(str(tmp_path)) / "configs" / "pipelines" / "dt" / "alg" / "v2.yaml"
    assert p == expected
