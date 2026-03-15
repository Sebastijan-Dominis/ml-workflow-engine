CONFIG_EXAMPLES_REGISTRY = {
"Model Specs":
"""problem: example_problem

segment:
  name: example_segment
  description: Example segment for model specs configuration.
version: v1
task:
  type: classification
  subtype: binary

target:
  name: is_example
  allowed_dtypes:
    - int64
  classes:
    count: 2
    positive_class: 1
    min_class_count: 200
  version: v1

segmentation:
  enabled: true
  include_in_model: false
  filters:
    - column: example_column_1
      op: eq
      value: Example Value 1
    - column: example_column_2
      op: eq
      value: Example Value 2

split:
  strategy: random
  stratify_by: is_example
  test_size: 0.2
  val_size: 0.12
  random_state: 42

algorithm: catboost
model_class: NameClassifier

pipeline:
  version: v1
  path: configs/pipelines/tabular/catboost/v1.yaml

feature_store:
  path: feature_store/
  feature_sets:
    - name: example_features
      version: v4
      data_format: parquet
      file_name: features.parquet
    - name: more_example_features
      version: v3
      data_format: parquet
      file_name: features.parquet

scoring:
  policy: adaptive_binary
  pr_auc_threshold: 0.1
class_weighting:
  policy: if_imbalanced
  imbalance_threshold: 0.1
  strategy: balanced

explainability:
  enabled: true
  top_k: 20
  methods:
    feature_importances:
      enabled: true
      type: PredictionValuesChange
    shap:
      enabled: true
      approximate: tree

data_type: tabular

model_specs_lineage:
  created_by: Name Surname
""",

"Search":
"""extends:
  - ../../../../configs/defaults/global.yaml
  - ../../../../configs/defaults/catboost.yaml
  - ../../../../configs/model_specs/example_problem/example_segment/v1.yaml

search:
  random_state: 42

  broad:
    iterations: 800
    n_iter: 50
    param_distributions:
      model:
        depth: [4, 6, 8, 10]
        learning_rate: [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
        l2_leaf_reg: [1.0, 3.0, 5.0, 7.0, 10.0, 15.0]
        random_strength: [1.0, 2.0, 5.0, 8.0, 10.0, 12.0]
        min_data_in_leaf: [1, 3, 5, 8, 10, 15, 20]
        border_count: [32, 64, 128, 254]
      ensemble:
        colsample_bylevel: [1.0]
        bagging_temperature: [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]

  narrow:
    enabled: true
    iterations: 800
    n_iter: 30
    param_configurations:
      model:
        depth:
          include: true
          offsets: [1, 2]
          low: 2
          high: 12
        learning_rate:
          include: true
          factors: [0.7, 0.85, 1.0, 1.15, 1.3]
          low: 0.003
          high: 0.5
          decimals: 5
        l2_leaf_reg:
          include: true
          factors: [0.7, 0.85, 1.0, 1.15, 1.3]
          low: 1.0
          high: 30.0
          decimals: 2
        random_strength:
          include: true
          factors: [0.7, 0.85, 1.0, 1.15, 1.3]
          low: 0.0
          high: 20.0
          decimals: 2
        min_data_in_leaf:
          include: true
          offsets: [2, 5, 10]
          low: 1
          high: 50
        border_count:
          include: true
      ensemble:
        bagging_temperature:
          include: true
          factors: [0.6, 0.8, 1.0, 1.2, 1.5]
          low: 0.0
          high: 5.0
          decimals: 3

search_lineage:
  created_by: Name Surname
""",

"Training":
"""extends:
  - ../../../../configs/defaults/global.yaml
  - ../../../../configs/defaults/catboost.yaml
  - ../../../../configs/model_specs/example_problem/example_segment/v1.yaml

training:
  iterations: 2500

training_lineage:
  created_by: Name Surname
"""
}
