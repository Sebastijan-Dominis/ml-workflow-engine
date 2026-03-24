"""Data config page for multi-page Dash app."""

from ml_service.frontend.configs.data.layout import build_layout
from ml_service.frontend.configs.data.callbacks import register_callbacks

def get_layout():
    """Return layout for data config page."""
    return build_layout()

def register(app):
    """Register callbacks for data config page."""
    register_callbacks(app)