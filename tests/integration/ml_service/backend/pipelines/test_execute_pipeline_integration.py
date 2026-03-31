"""Integration tests for `ml_service.backend.pipelines.execute_pipeline`.

These tests exercise the real subprocess invocation codepath by creating a
temporary test module under `tests/` and invoking it via
`execute_pipeline(... )` so that `python -m <module>` is executed.

Note: these tests create and remove transient files inside `tests/` and
are safe to run on both Windows and Linux CI agents.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import ml_service.backend.pipelines.execute_pipeline as ep
import pytest
from ml_service.backend.pipelines.execute_pipeline import execute_pipeline
from pydantic import BaseModel

pytestmark = pytest.mark.integration


class Payload(BaseModel):
    name: str | None = None
    flag: bool | None = None
    empty: str | None = None


def test_execute_pipeline_builds_cmd_and_returns_status(monkeypatch: Any) -> None:
    payload = Payload(name="abc", flag=True, empty="")

    captured: dict[str, Any] = {}

    def fake_run(cmd, capture_output, text, env, cwd):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(ep, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(ep, "EXIT_MEANING", {0: "SUCCESS"})

    res = ep.execute_pipeline("ml_service.pipelines.foo", payload, boolean_args=["flag"])  # type: ignore[arg-type]

    assert captured["cmd"][:3] == ["python", "-m", "ml_service.pipelines.foo"]
    # flags present and empty skipped
    assert "--name" in captured["cmd"] and "abc" in captured["cmd"]
    assert "--flag" in captured["cmd"] and "True" in captured["cmd"]
    assert "--empty" not in captured["cmd"]

    assert res["exit_code"] == 0
    assert res["status"] == "SUCCESS"
    assert res["stdout"] == "ok"
    assert res["stderr"] == ""


def _make_dummy_package(pkg_name: str, code: str) -> Path:
    base = Path("tests") / pkg_name
    base.mkdir(parents=True, exist_ok=False)
    (base / "__init__.py").write_text("")
    (base / "dummy_pipeline.py").write_text(code)
    return base


def _remove_dummy_package(base: Path) -> None:
    shutil.rmtree(base)


def test_execute_pipeline_runs_real_subprocess() -> None:
    """Create a transient module and run it via subprocess.

    The dummy module prints a JSON object containing the CLI args and exits
    with code ``0`` normally and ``2`` when ``--param1 fail`` is provided.
    This verifies argument marshalling, stdout capture and exit-code
    propagation from the subprocess invocation.
    """

    pkg_name = f"_integration_temp_pkg_{uuid4().hex}"
    code = dedent(
        """\
        from __future__ import annotations
        import argparse
        import json
        import sys

        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--param1", type=str, default="")
            parser.add_argument("--flag", type=str, default="False")
            args = parser.parse_args()
            print(json.dumps({"param1": args.param1, "flag": args.flag}))
            if args.param1 == "fail":
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
            param1: str | None = None
            flag: bool | None = None

        payload = LocalPayload(param1="ok", flag=True)
        res: dict[str, Any] = execute_pipeline(
            f"tests.{pkg_name}.dummy_pipeline", payload, boolean_args=["flag"]
        )

        assert isinstance(res, dict)
        assert res["exit_code"] == 0
        # ensure the module printed the JSON we expect
        assert '"param1": "ok"' in res["stdout"]
        # boolean flag should be present and represented as a string
        assert "True" in res["stdout"] or '"flag"' in res["stdout"]

        # Non-zero exit path
        payload2 = LocalPayload(param1="fail", flag=False)
        res2 = execute_pipeline(f"tests.{pkg_name}.dummy_pipeline", payload2, boolean_args=["flag"])  # type: ignore[arg-type]
        assert res2["exit_code"] != 0
    finally:
        if base.exists():
            _remove_dummy_package(base)
