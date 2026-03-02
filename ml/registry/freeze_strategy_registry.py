
"""Registry mapping feature data types to freeze strategy implementations."""

from ml.feature_freezing.freeze_strategies.tabular.strategy import \
    FreezeTabular
from ml.feature_freezing.freeze_strategies.time_series import FreezeTimeSeries

STRATEGIES = {
    "tabular": FreezeTabular,
    "time_series": FreezeTimeSeries,
}