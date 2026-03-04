import ast
import re
import sys
from pathlib import Path

ROOTS = [Path("ml"), Path("pipelines"), Path("scripts")]

SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
PASCAL_CASE_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
MODULE_RE = re.compile(r"^[a-z][a-z0-9_]*\.py$")

violations: list[str] = []


def check_module_name(file: Path):
    if file.name == "__init__.py":
        return

    if not MODULE_RE.match(file.name):
        violations.append(
            f"{file.as_posix()} -> module name '{file.name}' should be in snake_case"
        )


def check_ast(file: Path):
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except SyntaxError as e:
        violations.append(f"{file.as_posix()} -> SyntaxError: {e}")
        return

    for node in ast.walk(tree):
        # Function definitions (including async)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not SNAKE_CASE_RE.match(node.name):
                violations.append(
                    f"{file.as_posix()}:{node.lineno} "
                    f"-> function name '{node.name}' should be in snake_case"
                )

        # Class definitions
        elif isinstance(node, ast.ClassDef) and not PASCAL_CASE_RE.match(node.name):
            violations.append(
                f"{file.as_posix()}:{node.lineno} "
                f"-> class name '{node.name}' should be in PascalCase"
            )


def main():
    for root in ROOTS:
        if not root.exists():
            continue

        for file in root.rglob("*.py"):
            check_module_name(file)
            check_ast(file)

    if violations:
        print("\n".join(violations))
        sys.exit(1)
    else:
        print("No naming violations found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
