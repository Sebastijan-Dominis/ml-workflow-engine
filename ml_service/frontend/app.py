"""Main multi-page Dash app with collapsible sidebar and separate Home button."""

import os

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Import all pages
from ml_service.frontend.configs.data.page import get_layout as data_layout
from ml_service.frontend.configs.data.page import register as data_register
from ml_service.frontend.configs.features.page import get_layout as features_layout
from ml_service.frontend.configs.features.page import register as features_register
from ml_service.frontend.configs.modeling.page import get_layout as modeling_layout
from ml_service.frontend.configs.modeling.page import register as modeling_register
from ml_service.frontend.configs.pipeline_cfg.page import get_layout as pipeline_cfg_layout
from ml_service.frontend.configs.pipeline_cfg.page import register as pipeline_cfg_register
from ml_service.frontend.configs.promotion_thresholds.page import get_layout as promotion_layout
from ml_service.frontend.configs.promotion_thresholds.page import register as promotion_register
from ml_service.frontend.dir_viewer.page import get_layout as dir_viewer_layout
from ml_service.frontend.dir_viewer.page import register as dir_viewer_register
from ml_service.frontend.docs.page import get_layout as docs_layout
from ml_service.frontend.docs.page import register as docs_register
from ml_service.frontend.file_viewer.page import get_layout as viewer_layout
from ml_service.frontend.file_viewer.page import register as viewer_register
from ml_service.frontend.pipelines.page import get_layout as pipelines_layout
from ml_service.frontend.pipelines.page import register as pipelines_register
from ml_service.frontend.scripts.page import get_layout as scripts_layout
from ml_service.frontend.scripts.page import register as scripts_register

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Initialize Dash
app = dash.Dash(
    __name__,
    assets_folder=os.path.join(REPO_ROOT, "assets"),
    external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"],
    suppress_callback_exceptions=True
)
server = app.server

# Page mapping
PAGES = {
    "Data Config": data_layout,
    "Feature Config": features_layout,
    "Modeling Config": modeling_layout,
    "Pipeline Config": pipeline_cfg_layout,
    "Promotion Thresholds": promotion_layout,
    "Pipelines": pipelines_layout,
    "Scripts": scripts_layout,
    "Docs": docs_layout,
    "File Viewer": viewer_layout,
    "Directory Viewer": dir_viewer_layout,
}

# Register page callbacks
for register_func in [
    data_register,
    features_register,
    modeling_register,
    pipeline_cfg_register,
    promotion_register,
    pipelines_register,
    scripts_register,
    docs_register,
    viewer_register,
    dir_viewer_register,
]:
    register_func(app)

# Home page
def home_layout() -> dbc.Container:
    return dbc.Container(
        [
            html.H1(
                "ML Workflow - Homepage",
                style={"marginTop": "1rem", "fontWeight": "bold"}
            ),
            html.P(
                "Welcome to the ML Service Frontend Dashboard. "
                "Use the sidebar to navigate through different pages. "
                "Read the docs to understand what each element of each config does.",
                style={"fontSize": "1.5rem", "marginTop": "4rem", "maxWidth": "50%", "marginLeft": "auto", "marginRight": "auto"}
            ),
            html.Ul(
                [
                    html.Li([
                        html.Strong("Data Config:"),
                        html.Ul([
                            html.Li("Create and save data configurations."),
                            html.Li("Includes interim + processed configs."),
                            html.Li("Saves to: configs/data/{config_type}/{dataset_name}/{dataset_version}.yaml")
                        ])
                    ], style={"marginBottom": "1rem"}),
                    html.Li([
                        html.Strong("Feature Config:"),
                        html.Ul([
                            html.Li("Create and save feature set configurations."),
                            html.Li("Saves to the feature registry."),
                            html.Li("Registry path: configs/feature_registry/features.yaml")
                        ])
                    ], style={"marginBottom": "1rem"}),
                    html.Li([
                        html.Strong("Modeling Config:"),
                        html.Ul([
                            html.Li("Create and save modeling configurations."),
                            html.Li("Includes model specs, search, and training configs."),
                            html.Li("Saves to: configs/{config_type}/{problem}/{segment}/{version}.yaml")
                        ])
                    ], style={"marginBottom": "1rem"}),
                    html.Li([
                        html.Strong("Pipeline Config:"),
                        html.Ul([
                            html.Li("Create and save pipeline configurations."),
                            html.Li("Saves to: configs/pipelines/{data_type}/{algorithm}/{pipeline_version}.yaml")
                        ])
                    ], style={"marginBottom": "1rem"}),
                    html.Li([
                        html.Strong("Promotion Thresholds:"),
                        html.Ul([
                            html.Li("Create and save promotion thresholds."),
                            html.Li("Saves to the promotion thresholds registry."),
                            html.Li("Registry path: configs/promotion/thresholds.yaml")
                        ])
                    ], style={"marginBottom": "1rem"}),
                    html.Li([
                        html.Strong("Pipelines:"),
                        html.Ul([
                            html.Li("Run ML pipelines."),
                            html.Li("Includes all of the pipelines found in the pipeline/ directory."),
                            html.Li("Optional arguments have grey background, required arguments have white background."),
                        ])
                    ], style={"marginBottom": "1rem"}),
                    html.Li([
                        html.Strong("Scripts:"),
                        html.Ul([
                            html.Li("Run scripts."),
                            html.Li("Includes all of the scripts found in the scripts/ directory."),
                            html.Li("Optional arguments have grey background, required arguments have white background."),
                        ])
                    ], style={"marginBottom": "1rem"}),
                    html.Li([
                        html.Strong("Docs:"),
                        html.Ul([
                            html.Li("View documentation."),
                            html.Li("Includes all markdown files found in the docs/ directory."),
                            html.Li("Click links to navigate between docs."),
                            html.Li("Internal links to parts of the same markdown file don't work yet, but links to other markdown files do work."),
                            html.Li("Some md elements may render oddly, but all content should be there. Read directly from the md file if something looks off."),
                        ])
                    ], style={"marginBottom": "1rem"}),
                    html.Li([
                        html.Strong("File Viewer:"),
                        html.Ul([
                            html.Li("View contents of files in the repository."),
                            html.Li("Supports .yaml and .json files."),
                            html.Li("Useful for quickly checking .yaml and .json files without leaving the dashboard."),
                        ])
                    ], style={"marginBottom": "1rem"}),
                    html.Li([
                        html.Strong("Directory Viewer:"),
                        html.Ul([
                            html.Li("View directory structure from a chosen directory in the repository."),
                            html.Li("Specify the directory to view by entering a path relative to the repository root (e.g. 'configs/data')."),
                            html.Li("Input box allows to write and copy paths without leaving the page."),
                        ])
                    ], style={"marginBottom": "1rem"}),
                ],
                style={
                    "fontSize": "1.2rem",
                    "paddingTop": "5rem",
                    "textAlign": "left",
                    "marginLeft": "10%",
                    "marginBottom": "5rem",
                }
            )
        ],
        fluid=True,
        style={
            "minHeight": "100%",
            "paddingTop": "50px",
            "backgroundColor": "#8fa0d8",
            "textAlign": "center",
            "overflowY": "auto",
        }
    )

# Sidebar links (excluding Home)
def generate_page_links():
    ICONS = {
        "Data Config": "database",
        "Feature Config": "gear",
        "Modeling Config": "bar-chart",
        "Pipeline Config": "diagram-3",
        "Promotion Thresholds": "graph-up",
        "Pipelines": "play-circle",
        "Scripts": "terminal",
        "Docs": "book",
        "File Viewer": "eye",
        "Directory Viewer": "folder",
    }
    links = []
    for name in PAGES:
        icon = ICONS.get(name)
        content = [html.I(className=f"bi bi-{icon} me-2"), name] if icon else name
        links.append(
            dbc.NavLink(content, href=f"/{name.replace(' ', '_')}", id=f"nav-{name.replace(' ', '_')}", active="exact")
        )
    return links

# Main layout with separate Home button and collapsible sidebar
app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Separate Home button
                        dbc.Button(
                            "Home",
                            id="home-button",
                            href="/",
                            color="secondary",
                            className="mb-3",
                            style={"width": "100%"}
                        ),

                        # Sidebar toggle button (hamburger icon)
                        dbc.Button(
                            "☰",
                            id="sidebar-toggle",
                            color="primary",
                            className="mb-3",
                            n_clicks=0,
                            style={"width": "100%"}
                        ),

                        # Collapsible sidebar with page links
                        dbc.Collapse(
                            dbc.Nav(
                                generate_page_links(),
                                vertical=True,
                                pills=True,
                            ),
                            id="sidebar-collapse",
                            is_open=True,
                        ),
                    ],
                    width=2,
                    style={"position": "sticky", "marginTop": "5rem"}
                ),
                dbc.Col(
                    html.Div(
                        id="page-content-container",
                        style={
                            "padding": "20px",
                            "height": "98vh",
                            "overflowY": "auto",
                        }
                    ),
                    width=10,
                )
            ],
            style={
                "minHeight": "100vh",
                "backgroundColor": "#c1cbda",
                "overflow": "hidden"
            }
        ),
    ],
    fluid=True,
    style={
        "overflow": "hidden",
    }
)

# Callback to toggle sidebar collapse
@app.callback(
    Output("sidebar-collapse", "is_open", allow_duplicate=True),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_sidebar(n, is_open):
    if n:
        return not is_open
    return is_open

# Page routing
@app.callback(
    Output("page-content-container", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if not pathname or pathname == "/":
        return home_layout()
    page_name = pathname.lstrip("/").replace("_", " ")
    page_layout_func = PAGES.get(page_name)
    if page_layout_func:
        return page_layout_func()
    return dbc.Container(html.H2("404: Page not found", style={"textAlign": "center", "marginTop": "50px"}))

# Active link highlighting
@app.callback(
    [Output(f"nav-{name.replace(' ', '_')}", "active") for name in PAGES],
    Input("url", "pathname")
)
def update_active_links(pathname):
    if not pathname:
        pathname = "/"
    current_page = pathname.lstrip("/").replace("_", " ")
    return [name == current_page for name in PAGES]

# Callback to close sidebar when Home is clicked
@app.callback(
    Output("sidebar-collapse", "is_open", allow_duplicate=True),
    Input("home-button", "n_clicks"),
    State("sidebar-collapse", "is_open"),
    prevent_initial_call=True
)
def close_sidebar_on_home(n_clicks, is_open):
    if n_clicks:
        return False
    return is_open

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
