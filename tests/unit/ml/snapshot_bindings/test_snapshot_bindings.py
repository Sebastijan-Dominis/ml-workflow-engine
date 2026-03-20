"""Unit tests for ml.snapshot_bindings modules.

These tests cover validation helpers and the registry extraction function
and exercise success and failure paths for configuration validation.
"""
from __future__ import annotations

import importlib
from typing import Any

import pytest
from ml.exceptions import ConfigError
from ml.snapshot_bindings.config.models import SnapshotBinding


def _sample_registry_dict() -> dict[str, Any]:
    """Return a minimally valid registry mapping for tests."""
    return {
        "example_binding": {
            "datasets": {
                "my_dataset": {
                    "v1": {"snapshot": "dataset-snapshot-1"}
                }
            },
            "feature_sets": {
                "my_feature_set": {
                    "v1": {"snapshot": "feature-snapshot-1"}
                }
            },
        }
    }


def test_validate_snapshot_binding_none_raises() -> None:
    """validate_snapshot_binding must raise a ConfigError when given ``None``.

    This covers the defensive branch for missing configuration objects.
    """
    from ml.snapshot_bindings.validation.validate_snapshot_binding import (
        validate_snapshot_binding,
    )

    with pytest.raises(ConfigError) as exc:
        validate_snapshot_binding(None)
    assert "No snapshot binding configuration provided" in str(exc.value)


def test_validate_snapshot_binding_expect_dataset_failure() -> None:
    """When dataset bindings are expected but missing, a ConfigError is raised."""
    from ml.snapshot_bindings.validation.validate_snapshot_binding import (
        validate_snapshot_binding,
    )

    sb = SnapshotBinding()
    with pytest.raises(ConfigError) as exc:
        validate_snapshot_binding(sb, expect_dataset_bindings=True)
    assert "Expected dataset bindings" in str(exc.value)


def test_validate_snapshot_binding_expect_feature_set_failure() -> None:
    """When feature set bindings are expected but missing, a ConfigError is raised."""
    from ml.snapshot_bindings.validation.validate_snapshot_binding import (
        validate_snapshot_binding,
    )

    sb = SnapshotBinding()
    with pytest.raises(ConfigError) as exc:
        validate_snapshot_binding(sb, expect_feature_set_bindings=True)
    assert "Expected feature set bindings" in str(exc.value)


def test_validate_snapshot_binding_success_returns_model() -> None:
    """A valid SnapshotBinding instance is returned unmodified by the validator."""
    from ml.snapshot_bindings.validation.validate_snapshot_binding import (
        validate_snapshot_binding,
    )

    sb = SnapshotBinding()
    out = validate_snapshot_binding(sb)
    assert out is sb


def test_validate_snapshot_binding_registry_success_and_failure() -> None:
    """Test both successful registry validation and an invalid-registry path.

    The success path ensures proper model construction and nested values.
    The failure path ensures a malformed registry raises ``ConfigError``.
    """
    from ml.snapshot_bindings.validation.validate_snapshot_binding import (
        validate_snapshot_binding_registry,
    )

    # Success
    registry_raw = _sample_registry_dict()
    validated = validate_snapshot_binding_registry(registry_raw)
    # validated should behave like a mapping to SnapshotBinding
    sb = validated.get("example_binding")
    assert isinstance(sb, SnapshotBinding)
    assert sb.datasets["my_dataset"]["v1"].snapshot == "dataset-snapshot-1"

    # Failure: make an invalid registry that will fail pydantic validation
    with pytest.raises(ConfigError):
        validate_snapshot_binding_registry({"bad": {"datasets": "not-a-mapping"}})


def test_get_and_validate_snapshot_binding_happy_path(monkeypatch) -> None:
    """The extraction function loads the YAML registry and returns a validated binding.

    We patch the module-local loader used by the function so tests don't depend
    on filesystem fixtures.
    """
    # Import the module under test freshly to ensure no cached state
    mod = importlib.import_module("ml.snapshot_bindings.extraction.get_snapshot_binding")

    # Patch the loader used in that module (it imports `load_yaml` into its namespace)
    monkeypatch.setattr(
        mod,
        "load_yaml",
        lambda path: _sample_registry_dict(),
    )

    out = mod.get_and_validate_snapshot_binding(
        "example_binding", expect_dataset_bindings=True, expect_feature_set_bindings=True
    )
    assert isinstance(out, SnapshotBinding)
    assert out.feature_sets["my_feature_set"]["v1"].snapshot == "feature-snapshot-1"


def test_get_and_validate_snapshot_binding_missing_key_raises(monkeypatch) -> None:
    """If the requested key is absent in the registry, a ConfigError is raised."""
    mod = importlib.import_module("ml.snapshot_bindings.extraction.get_snapshot_binding")

    monkeypatch.setattr(mod, "load_yaml", lambda path: {})

    with pytest.raises(ConfigError):
        mod.get_and_validate_snapshot_binding("nope")
