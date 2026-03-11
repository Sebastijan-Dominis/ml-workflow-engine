# Validation Guarantees

## Config Validation

### Guarantees

- Type correctness (via Pydantic)
- Required fields present
- Enum constraints respected
- Structured schema compliance

### Does *Not* Guarantee

- Semantic correctness of business logic
- Task and algorithm suitability for a given problem type
- Optimal hyperparameter choices
- Optimal class weighting or target transformation

## Feature Store Validation

### Guarantees

- Required columns exist
- Dataset(s)' hashes at runtime match those from metadata
- Forbidden nulls enforced
- Cardinality constraints respected
- Row id present and included
- Required operator dependencies satisfied

### Does *Not* Guarantee

- Absence of data leakage
- Distribution stability
- Statistical sanity

## Search Validation

### Guarantees

- Algorithm specified is supported
- Scoring method is among supported scoring functions
- Search configuration schema is valid
- Class weighting allowed only for classification tasks
- Feature transformation allowed only for regression tasks
- `failure_management` directory structure enforced
- Best parameters originate from recorded search results
- Search metadata is persisted atomically

### Does *Not* Guarantee

- Statistical validity of CV splits
- Convergence to optimal hyperparameters
- Metrics suitability for business objective
- Absence of data leakage during CV
- Param distribution quality
- Compatibility between scoring function and specific algorithm beyong supported enum check

## Promotion Validation

### Guarantees

- Run IDs match across artifacts (`train_run_id` of the run that produced the candidate model matches `train_run_id` recorded in metadata of relevant evaluation and explainability runs)
- Artifacts (model, pipeline) consistent across runners - their hashes match each other, as well as runtime-computed ones
- Explainability artifacts exist (if supposed to) and runtime hash matches the one in metadata
- Required evaluation metrics exist
- Registry integrity preserved
- Archive integrity preserved

### Does *Not* Guarantee

- Business correctness of thresholds
- Production performance stability
- Fairness properties

## Runtime Validation

### Guarantees

- Environment snapshot captured
- Python version recorded
- Git commit recorded
- Hardware metadata captured

### Does *Not* Guarantee

- Bitwise reproducibility across machines
- GPU nondeterminism elimination
- Error-raising on mismatch (warnings get logged, but it was decided that runtime variance should not block the execution by default)