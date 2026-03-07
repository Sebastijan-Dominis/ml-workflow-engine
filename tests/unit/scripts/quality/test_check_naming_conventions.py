"""Unit tests for naming-convention quality checks."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.quality import check_naming_conventions as module

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_violations() -> None:
    """Reset global violation state between tests to avoid cross-test leakage."""
    module.violations.clear()


def test_check_module_name_flags_non_snake_case_filename(tmp_path: Path) -> None:
    """Record a violation when a Python module filename is not snake_case."""
    bad_module = tmp_path / "BadModule.py"
    bad_module.write_text("pass\n", encoding="utf-8")

    module.check_module_name(bad_module)

    assert len(module.violations) == 1
    assert "module name 'BadModule.py' should be in snake_case" in module.violations[0]


def test_check_ast_flags_invalid_public_function_and_class_names(tmp_path: Path) -> None:
    """Record violations for public function/class identifiers that break naming rules."""
    source = (
        "def BadFunction():\n"
        "    return 1\n\n"
        "class badclass:\n"
        "    pass\n\n"
        "def _private_helper():\n"
        "    return 2\n"
    )
    file = tmp_path / "module.py"
    file.write_text(source, encoding="utf-8")

    module.check_ast(file)

    assert len(module.violations) == 2
    assert "function name 'BadFunction' should be in snake_case" in module.violations[0]
    assert "class name 'badclass' should be in PascalCase" in module.violations[1]


def test_check_ast_records_syntax_errors(tmp_path: Path) -> None:
    """Record syntax errors instead of raising, preserving full scan behavior."""
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n    pass\n", encoding="utf-8")

    module.check_ast(broken)

    assert len(module.violations) == 1
    assert "SyntaxError" in module.violations[0]


def test_check_module_name_ignores_init_and_ignored_paths(tmp_path: Path) -> None:
    """Skip module-name checks for `__init__.py` and files under ignored folders."""
    init_file = tmp_path / "__init__.py"
    ignored_file = tmp_path / "tests" / "BadModule.py"
    ignored_file.parent.mkdir(parents=True)
    init_file.write_text("pass\n", encoding="utf-8")
    ignored_file.write_text("pass\n", encoding="utf-8")

    module.check_module_name(init_file)
    module.check_module_name(ignored_file)

    assert module.violations == []


def test_check_ast_skips_ignored_paths_and_private_class_names(tmp_path: Path) -> None:
    """Skip AST violations for ignored folders and private class/function identifiers."""
    ignored = tmp_path / "tests" / "module.py"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("def BadFunction():\n    return 1\n", encoding="utf-8")

    private_only = tmp_path / "private_only.py"
    private_only.write_text(
        "class _PrivateClass:\n"
        "    pass\n\n"
        "def __all__():\n"
        "    return []\n",
        encoding="utf-8",
    )

    module.check_ast(ignored)
    module.check_ast(private_only)

    assert module.violations == []


def test_check_ast_accepts_pascal_case_public_class_name(tmp_path: Path) -> None:
    """Avoid false positives for correctly named public classes."""
    good = tmp_path / "good_class.py"
    good.write_text("class GoodClass:\n    pass\n", encoding="utf-8")

    module.check_ast(good)

    assert module.violations == []


def test_is_ignored_handles_empty_ignore_parts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Skip empty ignore-folder entries without misclassifying files as ignored."""
    monkeypatch.setattr(module, "IGNORE_FOLDERS", [Path(".")])

    assert module.is_ignored(tmp_path / "sample.py") is False


def test_main_exits_zero_when_no_violations_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return zero exit status when scan completes without naming violations."""
    root = tmp_path / "ml"
    root.mkdir()
    good = root / "good_module.py"
    good.write_text("def valid_name():\n    return 1\n", encoding="utf-8")

    monkeypatch.setattr(module, "ROOTS", [root])
    monkeypatch.setattr(sys, "argv", ["check_naming_conventions.py"])

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 0
    assert "No naming violations found." in capsys.readouterr().out


def test_main_exits_nonzero_when_violations_found_via_cli_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return non-zero exit status and print violations when CLI files are provided."""
    bad = tmp_path / "BadModule.py"
    bad.write_text("def valid_name():\n    return 1\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["check_naming_conventions.py", str(bad)])

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1
    assert "module name 'BadModule.py' should be in snake_case" in capsys.readouterr().out


def test_main_skips_nonexistent_roots_and_still_exits_zero_when_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle missing roots gracefully while scanning existing roots."""
    existing_root = tmp_path / "scripts"
    existing_root.mkdir()
    (existing_root / "good_module.py").write_text("def ok_name():\n    return 1\n", encoding="utf-8")

    missing_root = tmp_path / "does_not_exist"
    monkeypatch.setattr(module, "ROOTS", [existing_root, missing_root])
    monkeypatch.setattr(sys, "argv", ["check_naming_conventions.py"])

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 0


def test_module_entrypoint_executes_main_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute module as script to cover the `if __name__ == "__main__"` path."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "good_module.py").write_text("def ok_name():\n    return 1\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_naming_conventions.py"])

    # Ensure runpy executes the module from a clean import state to avoid
    # RuntimeWarning about preloaded package modules.
    module_name = "scripts.quality.check_naming_conventions"
    existing_module = sys.modules.pop(module_name, None)

    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_module(module_name, run_name="__main__")
    finally:
        if existing_module is not None:
            sys.modules[module_name] = existing_module

    assert exc.value.code == 0
