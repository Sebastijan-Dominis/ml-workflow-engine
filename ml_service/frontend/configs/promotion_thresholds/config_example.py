EXAMPLE_CONFIG = """promotion_metrics:
  sets: [test]
  metrics: [mae, rmse]
  directions:
    mae: minimize
    rmse: minimize
thresholds:
  test:
    mae: 30
    rmse: 50
lineage:
  created_by: Name Surname
"""
