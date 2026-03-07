"""Unit tests for import-layer architecture checks."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


SCRIPT_PATH = Path("scripts/quality/check_import_layers.py")


def _run_script_as_main(script_path: Path) -> int:
    """Execute a script with ``__main__`` semantics and return exit code."""
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(script_path), run_name="__main__")
    code = exc_info.value.code
    return 0 if code is None else int(code)


def test_import_layers_fails_when_ml_imports_pipelines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a forbidden dependency where code under ``ml`` imports ``pipelines``."""
    script_path = SCRIPT_PATH.resolve()

    bad_file = tmp_path / "ml" / "bad_module.py"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("import pipelines.runners.train\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    exit_code = _run_script_as_main(script_path)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "ml must not import pipelines" in captured.out
    assert "ml/bad_module.py:1" in captured.out


def test_import_layers_fails_when_python_exists_in_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject Python files in ``configs`` because that tree must stay declarative-only."""
    script_path = SCRIPT_PATH.resolve()

    forbidden = tmp_path / "configs" / "bad.py"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("x = 1\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    exit_code = _run_script_as_main(script_path)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "configs should only contain yaml files, no python code" in captured.out
    assert "configs/bad.py" in captured.out


def test_import_layers_pass_for_valid_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return success when imports follow boundary rules and configs contains only YAML."""
    script_path = SCRIPT_PATH.resolve()

    good_file = tmp_path / "ml" / "good_module.py"
    good_file.parent.mkdir(parents=True, exist_ok=True)
    good_file.write_text("import os\n", encoding="utf-8")

    cfg = tmp_path / "configs" / "global.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("env: dev\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    exit_code = _run_script_as_main(script_path)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Import layer check passed." in captured.out


def test_import_layers_fails_on_deep_registry_catalog_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject deep imports into registry catalog internals outside package exports."""
    script_path = SCRIPT_PATH.resolve()

    file = tmp_path / "ml" / "consumer.py"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        "from ml.registries.catalogs.some_module import REGISTRY\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    exit_code = _run_script_as_main(script_path)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "use package exports instead of deep registry import" in captured.out
    assert "ml.registries.catalogs.some_module" in captured.out


def test_import_layers_fails_on_deep_registry_factory_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject deep imports into registry factory internals outside package exports."""
    script_path = SCRIPT_PATH.resolve()

    file = tmp_path / "ml" / "consumer.py"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        "import ml.registries.factories.internal_factory\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    exit_code = _run_script_as_main(script_path)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "use package exports instead of deep registry import" in captured.out
    assert "ml.registries.factories.internal_factory" in captured.out


def test_import_layers_fails_on_catalog_factory_cross_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject cross-imports between registry catalogs and factories to preserve separation."""
    script_path = SCRIPT_PATH.resolve()

    catalog_file = tmp_path / "ml" / "registries" / "catalogs" / "catalog_entry.py"
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    catalog_file.write_text("from ml.registries.factories import make_factory\n", encoding="utf-8")

    factory_file = tmp_path / "ml" / "registries" / "factories" / "factory_entry.py"
    factory_file.parent.mkdir(parents=True, exist_ok=True)
    factory_file.write_text("import ml.registries.catalogs\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    exit_code = _run_script_as_main(script_path)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "catalogs must not import factories" in captured.out
    assert "factories must not import catalogs" in captured.out
