# Promotion Thresholds

Promotion thresholds registry defines the **minimum performance thresholds** a trained model must meet in order to be promoted (for example from candidate → production).

Promotion rules act as a **safety gate** to prevent models with insufficient performance from being promoted.

Unlike some of the other configuration types, promotion rules are **centralized in a single configuration file** and selected dynamically during promotion evaluation.

## 1. Location

Promotion rules are defined in a single file:

```text
configs/promotion/thresholds.yaml
```

This file contains **nested promotion policies** for different tasks, datasets, and segments.

The promotion pipeline selects the appropriate policy at runtime using hierarchical lookup.

## 2. Configuration Structure

The configuration is organized hierarchically.

General structure:

```text
problem_type
  └─ segment
       └─ promotion_policy
```

Example:

```yaml
cancellation:

    global:

      promotion_metrics:
        sets: [test]
        metrics: [roc_auc, f1]

        directions:
          roc_auc: maximize
          f1: maximize

      thresholds:
        test:
          roc_auc: 0.75
          f1: 0.65

      lineage:
        created_by: "Sebastijan"
        created_at: "2026-02-27t23:53:30z"
```

Policy is selected based on problem type and segment at runtime.

## 3. Fields

### Promotion Metrics

Promotion checks are based on a configurable set of evaluation metrics.

Example:

```yaml
promotion_metrics:
  sets: [test]
  metrics: [roc_auc, f1]
```

#### Metric Sets

Metrics can be evaluated on different dataset splits:

| Set     | Description        |
| ------- | ------------------ |
| `train` | training dataset   |
| `val`   | validation dataset |
| `test`  | test dataset       |

### Metric Direction

Each metric must specify whether it should be maximized or minimized.

Example:

```yaml
directions:
  roc_auc: maximize
  log_loss: minimize
```

### Thresholds

Threshold values define the **minimum acceptable performance** for promotion.

Example:

```yaml
thresholds:
  test:
    roc_auc: 0.75
    f1: 0.65
```

A model must satisfy **all thresholds** to pass promotion.

### Lineage Metadata

Each promotion policy includes lineage metadata for traceability.

Example:

```yaml
lineage:
  created_by: "Sebastijan"
  created_at: "2026-02-27T23:53:30Z"
```

## 4. Validation

Promotion configurations are validated using the `PromotionThresholds` schema.

Validation ensures:
- directions exist for all metrics
- metric sets match threshold blocks
- threshold metrics match configured metrics

Invalid configurations raise a `ConfigError`.