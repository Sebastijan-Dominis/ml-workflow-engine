"""Unit tests for final CatBoost search-results assembly helper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.search.searchers.catboost import search_results_creator as module

pytestmark = pytest.mark.unit


def test_create_search_results_prefers_narrow_best_params_when_narrow_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use narrow-phase best params and include both broad+narrow phase payloads."""
    ctx = SimpleNamespace(
        best_params_1={"Model__depth": 6},
        best_params={"Model__depth": 7},
        broad_result={"best_score": 0.81, "search_phase": "broad"},
        narrow_result={"best_score": 0.83, "search_phase": "narrow"},
        require_narrow_disabled=False,
        require_best_params={"Model__depth": 7},
    )

    captured: dict[str, Any] = {}

    def _extract_model_params(best_pipeline_params: dict[str, Any]) -> dict[str, Any]:
        captured["best_pipeline_params"] = best_pipeline_params
        return {"depth": 7}

    monkeypatch.setattr(module, "extract_model_params", _extract_model_params)

    result = module.create_search_results(cast(Any, ctx))

    assert captured["best_pipeline_params"] == {"Model__depth": 7}
    assert result["best_pipeline_params"] == {"Model__depth": 7}
    assert result["best_model_params"] == {"depth": 7}
    assert result["phases"]["broad"] == {"best_score": 0.81, "search_phase": "broad"}
    assert result["phases"]["narrow"] == {"best_score": 0.83, "search_phase": "narrow"}


def test_create_search_results_uses_broad_params_and_omits_narrow_phase_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use broad-phase best params and do not emit narrow phase when disabled."""
    ctx = SimpleNamespace(
        best_params_1={"Model__depth": 6},
        best_params={"Model__depth": 7},
        broad_result={"best_score": 0.81, "search_phase": "broad"},
        narrow_result={"best_score": 0.83, "search_phase": "narrow"},
        require_narrow_disabled=True,
        require_best_params={"Model__depth": 7},
    )

    monkeypatch.setattr(module, "extract_model_params", lambda best_pipeline_params: {"depth": 6})

    result = module.create_search_results(cast(Any, ctx))

    assert result["best_pipeline_params"] == {"Model__depth": 6}
    assert result["best_model_params"] == {"depth": 6}
    assert result["phases"]["broad"] == {"best_score": 0.81, "search_phase": "broad"}
    assert "narrow" not in result["phases"]
