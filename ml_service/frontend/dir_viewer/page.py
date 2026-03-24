"""Directory Viewer page for the ML Service frontend."""

from .callbacks import register_callbacks
from .layout import build_layout


def get_layout():
    """Return the layout for the Directory Viewer page."""
    return build_layout()

def register(app):
    """Register callbacks for the Directory Viewer page."""
    register_callbacks(app)
