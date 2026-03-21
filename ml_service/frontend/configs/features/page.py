"""Features config page for multi-page Dash app."""

from ml_service.frontend.configs.features.callbacks import register_callbacks
from ml_service.frontend.configs.features.layout import build_layout


def get_layout():
    """Return layout for features page."""
    return build_layout()

def register(app):
    """Register callbacks for features page."""
    register_callbacks(app)
