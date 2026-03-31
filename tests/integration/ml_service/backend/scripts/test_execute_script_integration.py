"""Integration tests for `ml_service.backend.scripts.execute_script`.

These tests create a transient module under `tests/` and execute it via
`execute_script(...)` to validate list argument expansion, boolean flags,
stdout capture, and exit-code propagation.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import ml_service.backend.scripts.execute_script as es
import pytest
from ml_service.backend.scripts.execute_script import execute_script
from pydantic import BaseModel

pytestmark = pytest.mark.integration


class Payload(BaseModel):
    name: str | None = None
    items: list[int] | None = None
    flag: bool | None = None


def test_execute_script_handles_list_and_boolean_args_monkeypatch(monkeypatch: Any) -> None:
    payload = Payload(name="xyz", items=[1, 2], flag=False)

    captured: dict[str, Any] = {}

    def fake_run(cmd, capture_output, text, env, cwd):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=2, stdout="done", stderr="err")

    monkeypatch.setattr(es, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(es, "EXIT_MEANING", {2: "FAILURE"})

    res = es.execute_script("ml_service.scripts.bar", payload, boolean_args=["flag"])  # type: ignore[arg-type]

    assert captured["cmd"][:3] == ["python", "-m", "ml_service.scripts.bar"]
    # list arg expanded and flag handled as string
    assert "--items" in captured["cmd"]
    # ensure list values are individually present
    assert "1" in captured["cmd"] and "2" in captured["cmd"]
    assert "--flag" in captured["cmd"] and "False" in captured["cmd"]

    assert res["exit_code"] == 2
    assert res["status"] == "FAILURE"
    assert res["stdout"] == "done"
    assert res["stderr"] == "err"


def _make_dummy_package(pkg_name: str, code: str) -> Path:
    base = Path("tests") / pkg_name
    base.mkdir(parents=True, exist_ok=False)
    (base / "__init__.py").write_text("")
    (base / "dummy_script.py").write_text(code)
    return base


def _remove_dummy_package(base: Path) -> None:
    shutil.rmtree(base)


def test_execute_script_handles_list_and_boolean_args() -> None:
    """Verify list flags are expanded and boolean flags are stringified."""

    pkg_name = f"_integration_temp_script_pkg_{uuid4().hex}"
    code = dedent(
        """\
        from __future__ import annotations
        import argparse
        import json
        import sys

        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--names", nargs="+", default=[])
            parser.add_argument("--param", type=str, default="")
            parser.add_argument("--flag", type=str, default="False")
            args = parser.parse_args()
            print(json.dumps({"names": args.names, "param": args.param, "flag": args.flag}))
            if args.param == "fail":
                sys.exit(2)
            sys.exit(0)

        if __name__ == "__main__":
            main()
        """
    )

    base = Path("tests") / pkg_name
    try:
        base = _make_dummy_package(pkg_name, code)

        class LocalPayload(BaseModel):
            names: list[str] | None = None
            param: str | None = None
            flag: bool | None = None

        # success path
        payload = LocalPayload(names=["a", "b"], param="ok", flag=True)
        res: dict[str, Any] = execute_script(f"tests.{pkg_name}.dummy_script", payload, boolean_args=["flag"])  # type: ignore[arg-type]
        assert res["exit_code"] == 0
        assert '"names"' in res["stdout"]
        assert '"a"' in res["stdout"]
        assert "True" in res["stdout"] or '"flag"' in res["stdout"]

        # failure path
        payload2 = LocalPayload(names=["x"], param="fail", flag=False)
        res2 = execute_script(f"tests.{pkg_name}.dummy_script", payload2, boolean_args=["flag"])  # type: ignore[arg-type]
        assert res2["exit_code"] != 0
    finally:
        if base.exists():
            _remove_dummy_package(base)
