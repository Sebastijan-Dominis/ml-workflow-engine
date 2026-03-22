"""Scripts page for Dash multi-page app."""

from ml_service.frontend.scripts.callbacks import register_callbacks
from ml_service.frontend.scripts.layout import build_layout


def get_layout():
    """Returns the layout for the pipelines page."""
    return build_layout()


def register(app):
    """Registers callbacks for this page."""
    register_callbacks(app)
