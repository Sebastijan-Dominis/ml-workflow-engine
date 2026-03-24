"""Layout for the documentation viewer page."""

import os
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html

REPO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
DOCS_ROOT = REPO_ROOT / "docs"


def build_layout():
    """Returns the Dash layout for the docs page."""
    return dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    [
                        html.H1(
                            "Documentation",
                            style={"marginBottom": "45px", "textAlign": "center"},
                        ),
                        dcc.Markdown(
                            id="doc-content",
                            style={
                                "backgroundColor": "white",
                                "padding": "20px",
                                "borderRadius": "10px",
                                "height": "80vh",
                                "overflowY": "auto",
                            },
                        ),
                    ],
                    width=12,
                )
            ),
        ],
        fluid=True,
        style={
            "backgroundColor": "#8fa0d8",
            "minHeight": "100%",
            "paddingTop": "45px",
            "paddingBottom": "50px",
        },
    )
