"""Unit tests for hyperparameter search runner CLI orchestration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ml.exceptions import UserError
from pipelines.search import search as module

pytestmark = pytest.mark.unit


def test_parse_args_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse required args and preserve default optional search flags."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["search", "--problem", "no_show", "--segment", "global", "--version", "v1"],
    )

    args = module.parse_args()

    assert args.problem == "no_show"
    assert args.segment == "global"
    assert args.version == "v1"
    assert args.experiment_id is None
    assert args.env == "default"
    assert args.strict is True
    assert args.logging_level == "INFO"
    assert args.owner == "Sebastijan"
    assert args.clean_up_failure_management is True
    assert args.overwrite_existing is False


def test_main_returns_one_when_provided_experiment_dir_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail fast when user provides an explicit experiment id that does not exist."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            experiment_id="exp_missing",
            env="default",
            strict=True,
            logging_level="INFO",
            owner="Sebastijan",
            clean_up_failure_management=True,
            overwrite_existing=False,
        ),
    )

    code = module.main()

    assert code == 1


def test_main_returns_one_when_search_dir_has_files_and_no_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Protect existing search artifacts from accidental overwrite by default."""
    monkeypatch.chdir(tmp_path)

    search_dir = tmp_path / "experiments" / "adr" / "global" / "v1" / "exp_1" / "search"
    search_dir.mkdir(parents=True)
    (search_dir / "metadata.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="adr",
            segment="global",
            version="v1",
            experiment_id="exp_1",
            env="default",
            strict=True,
            logging_level="INFO",
            owner="Sebastijan",
            clean_up_failure_management=True,
            overwrite_existing=False,
        ),
    )
    monkeypatch.setattr(module, "setup_logging", lambda *args, **kwargs: None)

    code = module.main()

    assert code == 1


def test_main_runs_search_and_persists_outputs_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Execute happy-path search flow and persist resulting experiment outputs."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="cancellation",
            segment="city_hotel",
            version="v2",
            experiment_id=None,
            env="test",
            strict=False,
            logging_level="DEBUG",
            owner="CI",
            clean_up_failure_management=False,
            overwrite_existing=True,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T200000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))

    setup_log_paths: list[Path] = []
    monkeypatch.setattr(module, "setup_logging", lambda path, level: setup_log_paths.append(path))

    cfg = SimpleNamespace(algorithm=SimpleNamespace(value="CatBoost"))
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(module, "add_config_hash", lambda model_cfg: model_cfg)

    search_output = SimpleNamespace(
        search_results={"best_score": 0.81},
        feature_lineage=[SimpleNamespace(feature="f1")],
        pipeline_hash="pipe-hash",
        scoring_method="roc_auc",
        splits_info={"cv": 5},
    )
    searcher = SimpleNamespace(search=lambda model_cfg, strict, failure_management_dir: search_output)
    monkeypatch.setattr(module, "get_searcher", lambda key: searcher)

    persisted: dict[str, Any] = {}
    monkeypatch.setattr(module, "persist_experiment", lambda model_cfg, **kwargs: persisted.update(kwargs))

    cleanup_calls: list[tuple[Path, bool, str]] = []
    monkeypatch.setattr(
        module,
        "delete_failure_management_folder",
        lambda *, folder_path, cleanup, stage: cleanup_calls.append((folder_path, cleanup, stage)),
    )

    code = module.main()

    assert code == 0
    assert setup_log_paths[0].name == "search.log"
    assert persisted["experiment_id"] == "20260306T200000_abcdef01"
    assert persisted["search_results"] == {"best_score": 0.81}
    assert persisted["owner"] == "CI"
    assert persisted["pipeline_hash"] == "pipe-hash"
    assert persisted["scoring_method"] == "roc_auc"
    assert persisted["splits_info"] == {"cv": 5}
    assert cleanup_calls[0][1:] == (False, "search")


def test_main_maps_user_error_through_resolve_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Convert raised `UserError` into mapped CLI exit code via resolver."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="lead_time",
            segment="global",
            version="v1",
            experiment_id=None,
            env="default",
            strict=True,
            logging_level="INFO",
            owner="Sebastijan",
            clean_up_failure_management=True,
            overwrite_existing=False,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T200500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))
    monkeypatch.setattr(module, "setup_logging", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(algorithm=SimpleNamespace(value="CatBoost"))
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(module, "add_config_hash", lambda model_cfg: model_cfg)

    err = UserError("invalid search config")

    def _raise_searcher(_key: str) -> Any:
        raise err

    monkeypatch.setattr(module, "get_searcher", _raise_searcher)
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 44 if e is err else 99)

    code = module.main()

    assert code == 44


def test_main_maps_unexpected_error_through_resolve_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Map non-user exceptions through resolver while exercising exception-logging branch."""
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="lead_time",
            segment="global",
            version="v1",
            experiment_id=None,
            env="default",
            strict=True,
            logging_level="INFO",
            owner="Sebastijan",
            clean_up_failure_management=True,
            overwrite_existing=False,
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260307T150000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="8899aabbccddeeff"))
    monkeypatch.setattr(module, "setup_logging", lambda *args, **kwargs: None)

    cfg = SimpleNamespace(algorithm=SimpleNamespace(value="CatBoost"))
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(module, "add_config_hash", lambda model_cfg: model_cfg)

    err = RuntimeError("unexpected boom")

    def _raise_searcher(_key: str) -> Any:
        raise err

    monkeypatch.setattr(module, "get_searcher", _raise_searcher)
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 55 if e is err else 99)

    code = module.main()

    assert code == 55
