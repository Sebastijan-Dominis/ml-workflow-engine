import yaml
import pandas as pd
import logging
logger = logging.getLogger(__name__)
import hashlib, inspect

from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

from ml.logging_config import setup_logging

class FreezeTimeSeries:
    def freeze(self, config):
        pass # To be implemented in the future