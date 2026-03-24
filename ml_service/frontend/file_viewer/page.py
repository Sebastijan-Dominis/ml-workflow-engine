"""Page module for the file viewer page."""
from ml_service.frontend.file_viewer.callbacks import register_callbacks
from ml_service.frontend.file_viewer.layout import build_layout


def get_layout():
    return build_layout()

def register(app):
    register_callbacks(app)
