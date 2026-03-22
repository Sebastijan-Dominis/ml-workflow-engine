# Snapshot Bindings Registry

Snapshot bindings registry defines which snapshot of each specified dataset
and feature set will be used by a script that uses these configs.

It allows the use of specific snapshots, hence enabling reproducibility.

Unlike some of the other configuration types, snapshot bindings are **centralized in a single configuration file** and selected dynamically at runtime of a given pipeline (likely `freeze.py`,
`search.py`, or `train.py`).

## 1. Location

Snapshot bindings are defined in a single file:

```text
configs/snapshot_bindings_registry/bindings.yaml
```

This file contains all of the defined **snapshots of snapshot bindings**.

## 2. Configuration Structure

The configuration is organized hierarchically

General structure:

```text
snapshot_id
  └─ snapshot_bindings
```

Example:

```yaml
2026-03-20T02-54-47_61509023:
  datasets:
    hotel_bookings:
      v1:
        snapshot: 2026-03-17T03-26-25_4e817323
      v2:
        snapshot: 2026-03-17T03-26-27_c04ca84b
  feature_sets:
    booking_context_features:
      v1:
        snapshot: 2026-03-17T06-41-04_530962ff
      v2:
        snapshot: 2026-03-17T04-27-05_b9099899
      v3:
        snapshot: 2026-03-17T04-27-07_6ecc2061
      v4:
        snapshot: 2026-03-17T04-27-08_e45a7462
      v5:
        snapshot: 2026-03-17T04-27-09_72290e7c
      v6:
        snapshot: 2026-03-17T04-27-10_f299b9d2
      v7:
        snapshot: 2026-03-17T04-27-12_eba7203e
    channel_and_agent_features:
      v1:
        snapshot: 2026-03-17T04-27-24_d2b1a0df
      v2:
        snapshot: 2026-03-17T04-27-25_1bd99237
      v3:
        snapshot: 2026-03-17T04-27-27_e73d2e8c
    customer_history_features:
      v1:
        snapshot: 2026-03-17T04-27-13_47665433
      v2:
        snapshot: 2026-03-17T04-27-14_1881a909
      v3:
        snapshot: 2026-03-17T04-27-15_c7be8caf
      v4:
        snapshot: 2026-03-17T04-27-17_1c0339c3
    pricing_party_features:
      v1:
        snapshot: 2026-03-17T04-27-18_2e72ae25
      v2:
        snapshot: 2026-03-17T04-27-19_dd815223
      v3:
        snapshot: 2026-03-17T04-27-20_0592cbb0
      v4:
        snapshot: 2026-03-17T04-27-21_c2f019fb
      v5:
        snapshot: 2026-03-17T04-27-23_8ef8fcc4
    room_allocation_features:
      v1:
        snapshot: 2026-03-17T04-27-28_0368fd56
      v2:
        snapshot: 2026-03-17T04-27-29_e07095bc
```

Required snapshots are selected based on snapshot id at runtime. Extra dataset and feature set snapshots do not break the execution. Missing ones do.

## 3. Fields

### Dataset Snapshot Bindings

Dataset snapshot bindings follow the following nesting policy:

```text
datasets:
  └─ {dataset_name}
       └─ {dataset_version}
            └─ snapshot_information
```

Each nested layer specifies which snapshot should be used.

Example:

```yaml
datasets:
hotel_bookings:
    v1:
    snapshot: 2026-03-17T03-26-25_4e817323
    v2:
    snapshot: 2026-03-17T03-26-27_c04ca84b
```

### Feature Set Snapshot Bindings

Feature set snapshot bindings follow the following nesting policy:

```text
feature_sets:
  └─ {feature_set_name}
       └─ {feature_set_version}
            └─ snapshot_information
```

Each nested layer specifies which snapshot should be used.

Example:

```yaml
feature_sets:
    booking_context_features:
      v1:
        snapshot: 2026-03-17T06-41-04_530962ff
      v2:
        snapshot: 2026-03-17T04-27-05_b9099899
      v3:
        snapshot: 2026-03-17T04-27-07_6ecc2061
      v4:
        snapshot: 2026-03-17T04-27-08_e45a7462
      v5:
        snapshot: 2026-03-17T04-27-09_72290e7c
      v6:
        snapshot: 2026-03-17T04-27-10_f299b9d2
      v7:
        snapshot: 2026-03-17T04-27-12_eba7203e
    channel_and_agent_features:
      v1:
        snapshot: 2026-03-17T04-27-24_d2b1a0df
      v2:
        snapshot: 2026-03-17T04-27-25_1bd99237
      v3:
        snapshot: 2026-03-17T04-27-27_e73d2e8c
```

## 4. Validation

Snapshot bindings are validated using the `validate_snapshot_binding_registry` and
`validate_snapshot_binding` functions.

Validation ensures:
- the config file is formatted properly
- snapshot is not empty
- snapshot contains dataset or feature set bindings, if required

Invalid configurations raise a `ConfigError`.