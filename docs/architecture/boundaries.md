# Boundaries

## `pipelines/`

- CLI/orchestration only

## `ml/`

- reusable business/domain logic
- should not depend on pipelines

## `configs/`

- declarative YAML only
- no executable code
- no imports

## No deep imports into registry internals

- f.x. use `from ml.registries.catalogs import FEATURE_OPERATORS` instead of `from ml.registries.catalogs.feature_operators_catalog import FEATURE_OPERATORS`
- all catalogs should be listed within `ml.registries.catalogs.__init__.py`, and then imported from there
- this keeps the code cleaner and safer

## Registries and factories should stay independent

- catalogs should not import factories and vice versa
- f.x., a module within `ml.registries.catalogs` should not have this import line: `from ml.registries.factories import {something}`
- likewise, a module within `ml.registries.factories` should not have this import line: `from ml.registries.catalogs import {something}`

## New shared code goes into domain package first

- avoid placing shared code in `ml.utils`
- place it where it logically belongs, e.g. in `ml.runners`, `ml.modeling`, `ml.promotion`, etc.
- `ml.utils` should only contain code that is genuinely reusable across multiple different domains
- for instance, loading json and yaml files, getting the current git commit, and setting up a pipeline runner belong to `ml.utils`
- `get_trainer.py` is only used by trainer, so it does not belong in `ml.utils`; instead it belongs to `ml.runners.training.utils`
- feature lineage validation is useful across runners and search, but not data and feature freezing, so it belongs to `ml.modeling.validation`, rather than `ml.utils`