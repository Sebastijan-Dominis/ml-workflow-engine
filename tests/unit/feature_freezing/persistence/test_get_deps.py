"""Unit tests for dependency version collection helpers."""

import pytest
from ml.exceptions import RuntimeMLError
from ml.feature_freezing.persistence.get_deps import get_deps

pytestmark = pytest.mark.unit


def test_get_deps_returns_expected_dependency_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collect a complete dependency map with expected package name keys."""
    monkeypatch.setattr(
        "ml.feature_freezing.persistence.get_deps.get_pkg_version",
        lambda name: f"{name}-version",
    )

    deps = get_deps()

    assert deps == {
        "numpy": "numpy-version",
        "pandas": "pandas-version",
        "scikit_learn": "scikit-learn-version",
        "pyarrow": "pyarrow-version",
        "pydantic": "pydantic-version",
        "PyYAML": "PyYAML-version",
    }


def test_get_deps_wraps_version_lookup_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap package version lookup errors as RuntimeMLError for callers."""

    def _raise(name: str) -> str:
        raise ValueError("missing package")

    monkeypatch.setattr("ml.feature_freezing.persistence.get_deps.get_pkg_version", _raise)

    with pytest.raises(RuntimeMLError, match="Failed to get package versions"):
        get_deps()
