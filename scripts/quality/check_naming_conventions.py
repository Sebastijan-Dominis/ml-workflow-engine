"""Check naming conventions across the codebase."""
import ast
import re
import sys
from pathlib import Path

# Directories to check
ROOTS = [Path("ml"), Path("pipelines"), Path("scripts")]

# Regex patterns
SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
PASCAL_CASE_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
MODULE_RE = re.compile(r"^[a-z][a-z0-9_]*\.py$")

# Where violations will be stored
violations: list[str] = []

# Folder to ignore
IGNORE_FOLDERS = [Path("tests")]


def is_ignored(file: Path) -> bool:
    """Check if a file is inside an ignored folder.

    Args:
        file (Path): The path to the file to check.

    Returns:
        bool: True if the file is in an ignored folder, False otherwise.
    """
    file_parts = [part.lower() for part in file.parts]

    for ignored in IGNORE_FOLDERS:
        ignored_parts = [part.lower() for part in ignored.parts if part not in (".", "")]
        if not ignored_parts:
            continue

        window = len(ignored_parts)
        for idx in range(0, len(file_parts) - window + 1):
            if file_parts[idx : idx + window] == ignored_parts:
                return True

    return False


def check_module_name(file: Path):
    """Check if the module name (filename) follows snake_case convention.

    Args:
        file (Path): The path to the Python file to check.

    Returns:
        None: This function does not return a value but appends any naming violations to the global 'violations' list.
    """
    if file.name == "__init__.py" or is_ignored(file):
        return

    if not MODULE_RE.match(file.name):
        violations.append(
            f"{file.as_posix()} -> module name '{file.name}' should be in snake_case"
        )


def check_ast(file: Path):
    """Parse the file and check function and class naming conventions.

    Args:
        file (Path): The path to the Python file to check.

    Returns:
        None: This function does not return a value but appends any naming violations to the global 'violations' list.
    """
    if is_ignored(file):
        return

    try:
        tree = ast.parse(file.read_text(encoding="utf-8"))
    except SyntaxError as e:
        violations.append(f"{file.as_posix()} -> SyntaxError: {e}")
        return

    for node in ast.walk(tree):
        # Function definitions (including async)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip _private, __init__, __all__
            if node.name.startswith("_") or node.name in ("__init__", "__all__"):
                continue

            if not SNAKE_CASE_RE.match(node.name):
                violations.append(
                    f"{file.as_posix()}:{node.lineno} "
                    f"-> function name '{node.name}' should be in snake_case"
                )

        # Class definitions
        elif isinstance(node, ast.ClassDef):
            # Skip _private classes (common in tests)
            if node.name.startswith("_"):
                continue

            if not PASCAL_CASE_RE.match(node.name):
                violations.append(
                    f"{file.as_posix()}:{node.lineno} "
                    f"-> class name '{node.name}' should be in PascalCase"
                )


def main():
    """Main function to check naming conventions across the codebase.

    This script checks that:
    - Python module filenames are in snake_case.
    - Function names are in snake_case (with exceptions for private and special methods).
    - Class names are in PascalCase (with exceptions for private classes).
    It traverses the specified directories, parses Python files, and collects any naming violations.
    Finally, it reports all violations and exits with an appropriate status code.

    Returns:
        None: This function does not return a value but exits the program with a status code."""
    files_to_check: list[Path] = []

    if len(sys.argv) > 1:
        files_to_check = [Path(f) for f in sys.argv[1:]]
    else:
        for root in ROOTS:
            if root.exists():
                files_to_check.extend(root.rglob("*.py"))

    for file in files_to_check:
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
