
from ml_service.backend.configs.features.utils.paths import get_registry_path


def test_get_registry_path(tmp_path):
    repo_root = tmp_path
    p = get_registry_path(repo_root)
    expected = repo_root / "configs" / "feature_registry" / "features.yaml"
    assert p == expected
