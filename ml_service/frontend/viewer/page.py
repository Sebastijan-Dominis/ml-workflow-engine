"""Page module for the file viewer page."""
from ml_service.frontend.viewer.callbacks import register_callbacks
from ml_service.frontend.viewer.layout import build_layout


def get_layout():
    return build_layout()

def register(app):
    register_callbacks(app)
