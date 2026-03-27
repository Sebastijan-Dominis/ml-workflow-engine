"""Module responsible for preparing features for post-promotion pipelines (monitoring and inference)."""
import argparse
from pathlib import Path

from ml.config.loader import load_and_validate_config
from ml.features.loading.resolve_feature_snapshots import resolve_feature_snapshots
from ml.metadata.validation.runners.training import validate_training_metadata
from ml.metadata.validation.search.search import validate_search_record
from ml.post_promotion.shared.classes.function_returns import PrepareFeaturesReturn
from ml.promotion.config.registry_entry import RegistryEntry
from ml.utils.loaders import load_json


def prepare_features(
    *,
    args: argparse.Namespace,
    model_metadata: RegistryEntry,
    snapshot_bindings_id: str | None = None
) -> PrepareFeaturesReturn:
    # Lazy import to avoid circular dependencies
    from ml.features.loading.features_and_target import load_features_and_target

    experiment_id = model_metadata.experiment_id
    train_run_id = model_metadata.train_run_id

    experiment_path = Path("experiments") / args.problem / args.segment / model_metadata.model_version / experiment_id

    training_metadata_file = experiment_path / "training" / train_run_id / "metadata.json"
    training_metadata = validate_training_metadata(load_json(training_metadata_file))

    feature_sets = training_metadata.lineage.feature_lineage

    if snapshot_bindings_id is None:
        snapshot_binding = training_metadata.lineage.feature_lineage
        resolved_snapshots = resolve_feature_snapshots(
            feature_store_path=Path("feature_store"),
            feature_sets=feature_sets,
            snapshot_binding = snapshot_binding
        )
    else:
        resolved_snapshots = resolve_feature_snapshots(
            feature_store_path=Path("feature_store"),
            feature_sets=feature_sets,
            snapshot_binding_key=snapshot_bindings_id
        )

    search_metadata = validate_search_record(
        load_json(experiment_path / "search" / "metadata.json")
    )

    model_version = search_metadata.metadata.version
    env = search_metadata.metadata.env

    model_cfg = load_and_validate_config(
        path=Path("configs") / "search" / args.problem / args.segment / f"{model_version}.yaml",
        search_dir=None,
        cfg_type="search",
        env=env
    )


    X, y, _, entity_key = load_features_and_target(
        model_cfg,
        snapshot_selection=resolved_snapshots,
        drop_entity_key=False, # could be useful for joins
        strict=True
    )

    return PrepareFeaturesReturn(
        features=X,
        entity_key=entity_key,
        feature_lineage=training_metadata.lineage.feature_lineage,
        target=y
    )
