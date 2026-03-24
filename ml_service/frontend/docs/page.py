"""Docs page for Dash multi-page app."""

from .callbacks import register_callbacks
from .layout import build_layout


def get_layout():
    """Returns the layout for the docs page."""
    return build_layout()


def register(app):
    """Registers callbacks for this page."""
    register_callbacks(app)
