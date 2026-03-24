"""A Dash app for interactively viewing file contents on the server filesystem."""
import dash_ace
import dash_bootstrap_components as dbc
from dash import html

PAGE_PREFIX = "/file_viewer"

def build_layout():
    return dbc.Container(
        [
            dbc.Col(
                [
                    html.H1("File Viewer", style={"textAlign": "center"}),

                    dbc.Label(
                        "Enter file path relative to repo root:",
                        html_for=f"{PAGE_PREFIX}-path-input",
                        style={"fontSize": "1.2rem"}
                    ),

                    dbc.Input(
                        id=f"{PAGE_PREFIX}-path-input",
                        placeholder="e.g. feature_store\\booking_context_features\\v1\\2026-03-24T03-40-09_04b28db7\\metadata.json",
                        type="text",
                        style={"fontSize": "1.2rem"},
                    ),

                    dbc.Button(
                        "Load",
                        id=f"{PAGE_PREFIX}-load-btn",
                        color="primary",
                        style={"width": "120px"},
                    ),

                    html.Div(id=f"{PAGE_PREFIX}-status"),

                    dash_ace.DashAceEditor(
                        id=f"{PAGE_PREFIX}-viewer",
                        mode="yaml",
                        theme="github",
                        value="",
                        readOnly=True,
                        fontSize=16,
                        height="700px",
                        setOptions={"showLineNumbers": True},
                        style={"marginTop": "20px"},
                        placeholder="File contents will be displayed here in YAML/JSON format after loading.",
                    ),
                ],
                style={
                    "alignItems": "center",
                    "display": "flex",
                    "flexDirection": "column",
                    "width": "50%",
                    "margin": "0 auto",
                    "gap": "1rem"
                }
            )
        ],
        fluid=True,
        style={
            "backgroundColor": "#8fa0d8",
            "minHeight": "100%",
            "paddingTop": "45px",
            "paddingBottom": "50px",
        },
    )
