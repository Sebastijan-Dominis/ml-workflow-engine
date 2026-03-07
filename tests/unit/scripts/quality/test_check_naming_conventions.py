"""Unit tests for naming-convention quality checks."""

from __future__ import annotations

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
