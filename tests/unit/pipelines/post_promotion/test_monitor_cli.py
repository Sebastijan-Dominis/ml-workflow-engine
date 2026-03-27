import importlib
from types import SimpleNamespace


def test_pipelines_monitor_main_success(monkeypatch):
    mod = importlib.import_module("pipelines.post_promotion.monitor")

    args = SimpleNamespace(problem="p", segment="s", inference_run_id="latest", logging_level="INFO")
    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(mod, "setup_logging", lambda *a, **kw: None)

    from ml.promotion.config.registry_entry import (
        RegistryArtifacts,
        RegistryEntry,
        RegistryEntryMetrics,
    )

    reg = RegistryEntry(
        experiment_id="e",
        train_run_id="t",
        eval_run_id="x",
        explain_run_id="r",
        model_version="mv",
        artifacts=RegistryArtifacts(model_hash="mh", model_path="mp"),
        feature_lineage=[],
        metrics=RegistryEntryMetrics(train={}, val={}, test={}),
        git_commit="c",
    )

    monkeypatch.setattr(mod, "get_promotion_metrics_info", lambda a: None)
    monkeypatch.setattr(mod, "get_model_registry_info", lambda a: SimpleNamespace(prod_meta=reg, stage_meta=reg))
    monkeypatch.setattr(mod, "execute_monitoring", lambda **kwargs: SimpleNamespace(drift_results={"a": 0}, performance_results={"m": {"v": 1}}, model_version="mv"))
    monkeypatch.setattr(mod, "prepare_metadata", lambda **kwargs: {})

    # Avoid invoking deep comparison logic; stub it out so metadata writing path is exercised.
    monkeypatch.setattr(mod, "compare_production_and_staging_performance", lambda p, s: {"cmp": 0})

    saved = []
    monkeypatch.setattr(mod, "save_metadata", lambda metadata, target_dir: saved.append((metadata, target_dir)))

    rc = mod.main()

    assert rc == 0
    assert saved


def test_pipelines_monitor_main_no_models(monkeypatch):
    mod = importlib.import_module("pipelines.post_promotion.monitor")

    args = SimpleNamespace(problem="p", segment="s", inference_run_id="latest", logging_level="INFO")
    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(mod, "setup_logging", lambda *a, **kw: None)

    monkeypatch.setattr(mod, "get_promotion_metrics_info", lambda a: None)
    monkeypatch.setattr(mod, "get_model_registry_info", lambda a: SimpleNamespace(prod_meta=None, stage_meta=None))

    monkeypatch.setattr(mod, "resolve_exit_code", lambda e: 42)

    rc = mod.main()

    assert rc == 42
