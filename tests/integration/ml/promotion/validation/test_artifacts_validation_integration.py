from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import ml.promotion.validation.artifacts as mod
import pytest

pytestmark = pytest.mark.integration


def test_validate_artifacts_consistency_happy_path(tmp_path: Path, monkeypatch: Any) -> None:
    # Create fake model and pipeline files
    model_file = tmp_path / "model.bin"
    model_file.write_text("model")
    pipeline_file = tmp_path / "pipeline.joblib"
    pipeline_file.write_text("pipeline")

    # Build simple metadata objects expected by the validator
    artifacts = SimpleNamespace(
        model_path=str(model_file),
        model_hash="deadbeef",
        pipeline_path=str(pipeline_file),
        pipeline_hash="pdeadbeef",
    )

    run_identity = SimpleNamespace(status="success")
    training_meta = SimpleNamespace(run_identity=run_identity, artifacts=artifacts)
    evaluation_meta = SimpleNamespace(run_identity=run_identity, artifacts=artifacts)
    explain_meta = SimpleNamespace(run_identity=run_identity, artifacts=artifacts)

    runners_metadata = SimpleNamespace(
        training_metadata=training_meta,
        evaluation_metadata=evaluation_meta,
        explainability_metadata=explain_meta,
    )

    # Fake hashing to match expected hashes
    def fake_hash_artifact(p: Path) -> str:
        if p.name == "model.bin":
            return "deadbeef"
        if p.name == "pipeline.joblib":
            return "pdeadbeef"
        return "x"

    monkeypatch.setattr(mod, "hash_artifact", fake_hash_artifact)

    # Should not raise
    mod.validate_artifacts_consistency(cast(mod.RunnersMetadata, runners_metadata))
