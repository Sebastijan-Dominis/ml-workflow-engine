"""E2E tests that exercise `execute_script` by creating a transient module.

These tests write a small module under `tests/` and call the real
`execute_script` helper which uses `python -m` to run the module as a subprocess.
"""

import contextlib
import shutil
import uuid
from pathlib import Path

import ml_service.backend.scripts.execute_script as exec_mod
from pydantic import BaseModel


def _make_temp_module() -> tuple[Path, str]:
    pkg_name = f"_tmp_exec_pkg_{uuid.uuid4().hex[:8]}"
    pkg_dir = Path("tests") / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=False)
    # create __init__ to make it a package
    (pkg_dir / "__init__.py").write_text("")
    # script that optionally fails when --fail flag is present
    script = (
        'import sys\n'
        'def main():\n'
        '    if "--fail" in sys.argv:\n'
        '        print("FAIL", flush=True)\n'
        '        sys.exit(3)\n'
        '    print("SCRIPT_RUN_OK", flush=True)\n'
        '    sys.exit(0)\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    mod_name = "runme"
    (pkg_dir / f"{mod_name}.py").write_text(script)
    return pkg_dir, f"tests.{pkg_name}.{mod_name}"


def test_execute_script_subprocess_success() -> None:
    pkg_dir, module_path = _make_temp_module()
    try:
        class Dummy(BaseModel):
            pass

        res = exec_mod.execute_script(module_path=module_path, payload=Dummy(), boolean_args=None)
        assert res["exit_code"] == 0
        assert "SCRIPT_RUN_OK" in res["stdout"]
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(pkg_dir)


def test_execute_script_subprocess_failure() -> None:
    pkg_dir, module_path = _make_temp_module()
    try:
        class DummyFail(BaseModel):
            fail: bool | None = True

        # ask for boolean flag 'fail' to be present
        res = exec_mod.execute_script(module_path=module_path, payload=DummyFail(), boolean_args=["fail"])
        assert res["exit_code"] != 0
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(pkg_dir)
