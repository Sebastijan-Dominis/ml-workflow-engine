"""Check import layers and dependencies across the codebase to enforce architectural boundaries.

Rules:
1) ml is reusable business/domain logic only, should not depend on pipelines
2) registries package internals should be imported via package exports only
   Allowed:
    # from ml.registries.catalogs import X
    # from ml.registries.factories import Y
    # from ml.registries import Z
Disallowed deep imports:
    # from ml.registries.catalogs.some_module import X
    # from ml.registries.factories.some_module import Y
3) catalogs and factories should stay independent (no cross-imports)
4) configs is declarative yaml only, should not import any code (including other configs), no executable code allowed - only yaml files
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOTS = [Path("ml"), Path("pipelines"), Path("scripts")]
CONFIGS_ROOT = Path("configs")

IMPORT_RE = re.compile(r"^\s*(from|import)\s+([a-zA-Z0-9_\.]+)")


def _scan_python_file(file: Path) -> list[str]:
    """Collect import-layer violations in a single Python file.

    Args:
        file: Python file to scan.

    Returns:
        list[str]: Violations with file and line context.
    """
    rel = file.as_posix()
    text = file.read_text(encoding="utf-8", errors="ignore")

    file_violations: list[str] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        m = IMPORT_RE.match(line)
        if not m:
            continue

        mod = m.group(2)

        # 1) ml is reusable business/domain logic only, should not depend on pipelines
        if rel.startswith("ml/") and (mod == "pipelines" or mod.startswith("pipelines.")):
            file_violations.append(f"{rel}:{idx} -> ml must not import pipelines ({mod})")

        # 2) registries package internals should be imported via package exports only
        #    Allowed:
        #      from ml.registries.catalogs import X
        #      from ml.registries.factories import Y
        #      from ml.registries import Z
        #    Disallowed deep imports:
        #      from ml.registries.catalogs.some_module import X
        #      from ml.registries.factories.some_module import Y
        if mod.startswith("ml.registries.catalogs.") or mod.startswith("ml.registries.factories."):
            file_violations.append(
                f"{rel}:{idx} -> use package exports instead of deep registry import ({mod})"
            )

        # 3) catalogs and factories should stay independent (no cross-imports)
        if rel.startswith("ml/registries/catalogs/") and (
            mod == "ml.registries.factories" or mod.startswith("ml.registries.factories.")
        ):
            file_violations.append(f"{rel}:{idx} -> catalogs must not import factories ({mod})")

        if rel.startswith("ml/registries/factories/") and (
            mod == "ml.registries.catalogs" or mod.startswith("ml.registries.catalogs.")
        ):
            file_violations.append(f"{rel}:{idx} -> factories must not import catalogs ({mod})")

    return file_violations


def collect_violations() -> list[str]:
    """Scan project roots and return all import-layer violations."""
    violations: list[str] = []

    for root in ROOTS:
        if not root.exists():
            continue

        for file in root.rglob("*.py"):
            violations.extend(_scan_python_file(file))

    # 4) configs is declarative yaml only; Python code in this tree is a violation.
    if CONFIGS_ROOT.exists():
        for file in CONFIGS_ROOT.rglob("*.py"):
            violations.append(
                f"{file.as_posix()} -> configs should only contain yaml files, no python code"
            )

    return violations


def main() -> int:
    """Run import-layer checks and report violations.

    Returns:
        int: Exit code, where ``0`` means success and ``1`` means violations found.
    """
    violations = collect_violations()

    if violations:
        print("Import layer violations found:")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print("Import layer check passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
