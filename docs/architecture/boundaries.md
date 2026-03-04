# Boundaries

## `pipelines/`

- CLI/orchestration only

## `ml/`

- reusable business/domain logic
- should not depend on pipelines

## `configs/`

- delarative YAML only
- no executable code
- no imports

## No deep imports into registry internals

- f.x. use `from ml.registries.catalogs` instead of `from ml.registries.catalogs.module_name`

## New shared code goes into domain package first

- avoid placing shared code in `utils/`