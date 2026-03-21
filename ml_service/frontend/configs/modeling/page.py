"""Modeling config page for Dash multi-page app."""

from ml_service.frontend.configs.modeling.callbacks import register_callbacks
from ml_service.frontend.configs.modeling.layout import build_layout


def get_layout():
    """Returns the layout for the modeling page."""
    return build_layout()

def register(app):
    """Register callbacks for this page."""
    register_callbacks(app)
