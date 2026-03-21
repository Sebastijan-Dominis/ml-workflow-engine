"""Pipeline config page for multi-page Dash app."""

from ml_service.frontend.configs.pipeline_cfg.callbacks import register_callbacks
from ml_service.frontend.configs.pipeline_cfg.layout import build_layout


def get_layout():
    """Return layout for pipeline config page."""
    return build_layout()

def register(app):
    """Register callbacks for pipeline config page."""
    register_callbacks(app)
