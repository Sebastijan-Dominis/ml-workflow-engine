"""Unit tests for explainability runner CLI orchestration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ml.exceptions import ConfigError
from pipelines.runners import explain as module

pytestmark = pytest.mark.unit


def test_parse_args_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse required args and preserve defaults for optional explainability flags."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["explain", "--problem", "no_show", "--segment", "global", "--version", "v1"],
    )

    args = module.parse_args()

    assert args.problem == "no_show"
    assert args.segment == "global"
    assert args.version == "v1"
    assert args.env == "default"
    assert args.strict is True
    assert args.experiment_id == "latest"
    assert args.train_id == "latest"
    assert args.logging_level == "INFO"
    assert args.top_k is None


def test_main_returns_resolved_code_when_snapshot_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map experiment/training snapshot resolution failures through exit resolver."""
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            env="default",
            strict=True,
            experiment_id="latest",
            train_id="latest",
            logging_level="INFO",
            top_k=None,
        ),
    )
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)

    err = RuntimeError("missing snapshots")
    monkeypatch.setattr(module, "get_snapshot_path", lambda snapshot_id, parent_dir: (_ for _ in ()).throw(err))
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 35 if e is err else 99)

    code = module.main()

    assert code == 35


def test_main_maps_config_error_when_explainability_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail through resolver when explainability is disabled in model config."""
    experiment_dir = tmp_path / "experiments" / "no_show" / "global" / "v1" / "exp_1"
    train_dir = experiment_dir / "training" / "train_1"
    train_dir.mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            env="default",
            strict=True,
            experiment_id="exp_1",
            train_id="train_1",
            logging_level="INFO",
            top_k=None,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T190000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(
        module,
        "get_snapshot_path",
        lambda snapshot_id, parent_dir: experiment_dir if parent_dir.name == "v1" else train_dir,
    )
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: None)

    cfg = SimpleNamespace(explainability=SimpleNamespace(enabled=False, top_k=10))
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(
        module,
        "resolve_exit_code",
        lambda e: 62 if isinstance(e, ConfigError) else 99,
    )

    code = module.main()

    assert code == 62


def test_main_runs_explainability_with_config_top_k_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use config-defined `top_k` when CLI override is omitted and persist outputs."""
    experiment_dir = tmp_path / "experiments" / "adr" / "global" / "v1" / "exp_2"
    train_dir = experiment_dir / "training" / "train_2"
    train_dir.mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="adr",
            segment="global",
            version="v1",
            env="test",
            strict=False,
            experiment_id="exp_2",
            train_id="train_2",
            logging_level="DEBUG",
            top_k=None,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T190500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(
        module,
        "get_snapshot_path",
        lambda snapshot_id, parent_dir: experiment_dir if parent_dir.name == "v1" else train_dir,
    )

    log_paths: list[Path] = []
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: log_paths.append(path))

    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(name="CatBoost"),
        explainability=SimpleNamespace(enabled=True, top_k=37),
    )
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(module, "add_config_hash", lambda model_cfg: model_cfg)

    monkeypatch.setattr(module, "validate_lineage_integrity", lambda train_dir_arg, model_cfg: None)
    monkeypatch.setattr(module, "validate_reproducibility", lambda runtime_path: None)
    monkeypatch.setattr(module, "validate_pipeline_cfg", lambda metadata_path, model_cfg: "pcfg")
    monkeypatch.setattr(module, "validate_model_and_pipeline", lambda train_dir_arg: SimpleNamespace(model_path="m"))

    explain_calls: list[dict[str, Any]] = []
    explain_output = SimpleNamespace(
        explainability_metrics={"gain": {"f1": 1.0}},
        feature_lineage=[SimpleNamespace(feature="f1")],
    )

    class _DummyExplainer:
        def explain(self, *, model_cfg: Any, train_dir: Path, top_k: int) -> Any:
            explain_calls.append({"top_k": top_k, "train_dir": train_dir})
            return explain_output

    monkeypatch.setattr(module, "get_explainer", lambda key: _DummyExplainer())

    persisted: dict[str, Any] = {}
    monkeypatch.setattr(
        module,
        "persist_explainability_run",
        lambda **kwargs: persisted.update(kwargs),
    )

    code = module.main()

    assert code == 0
    assert log_paths[0].name == "explainability.log"
    assert explain_calls == [{"top_k": 37, "train_dir": train_dir}]
    assert persisted["explain_run_id"] == "20260306T190500_00112233"
    assert persisted["train_run_id"] == "train_2"
    assert persisted["top_k"] == 37


def test_main_prefers_cli_top_k_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use CLI `--top-k` value instead of config default when provided."""
    experiment_dir = tmp_path / "experiments" / "adr" / "global" / "v2" / "exp_3"
    train_dir = experiment_dir / "training" / "train_3"
    train_dir.mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="adr",
            segment="global",
            version="v2",
            env="default",
            strict=True,
            experiment_id="exp_3",
            train_id="train_3",
            logging_level="INFO",
            top_k=12,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T191000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="8899aabbccddeeff"))
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(
        module,
        "get_snapshot_path",
        lambda snapshot_id, parent_dir: experiment_dir if parent_dir.name == "v2" else train_dir,
    )
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: None)

    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(name="CatBoost"),
        explainability=SimpleNamespace(enabled=True, top_k=99),
    )
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(module, "add_config_hash", lambda model_cfg: model_cfg)
    monkeypatch.setattr(module, "validate_lineage_integrity", lambda train_dir_arg, model_cfg: None)
    monkeypatch.setattr(module, "validate_reproducibility", lambda runtime_path: None)
    monkeypatch.setattr(module, "validate_pipeline_cfg", lambda metadata_path, model_cfg: "pcfg")
    monkeypatch.setattr(module, "validate_model_and_pipeline", lambda train_dir_arg: SimpleNamespace(model_path="m"))

    captured_top_k: dict[str, int] = {}

    class _DummyExplainer:
        def explain(self, *, model_cfg: Any, train_dir: Path, top_k: int) -> Any:
            captured_top_k["value"] = top_k
            return SimpleNamespace(explainability_metrics={}, feature_lineage=[])

    monkeypatch.setattr(module, "get_explainer", lambda key: _DummyExplainer())
    monkeypatch.setattr(module, "persist_explainability_run", lambda **kwargs: None)

    code = module.main()

    assert code == 0
    assert captured_top_k["value"] == 12
