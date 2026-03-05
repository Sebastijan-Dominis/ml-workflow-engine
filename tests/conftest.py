""""Pytest configuration file for the hotel management project. This file sets up the testing environment by adding the project root directory to the Python path, allowing tests to import modules from the project without issues. It ensures that all tests can access the necessary code and resources from the project when run."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
