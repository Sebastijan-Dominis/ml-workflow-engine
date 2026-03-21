"""Promotion thresholds page for multi-page Dash app."""

from ml_service.frontend.configs.promotion_thresholds.callbacks import register_callbacks
from ml_service.frontend.configs.promotion_thresholds.layout import build_layout


def get_layout():
    """Return layout for promotion thresholds page."""
    return build_layout()

def register(app):
    """Register callbacks for promotion thresholds page."""
    register_callbacks(app)
