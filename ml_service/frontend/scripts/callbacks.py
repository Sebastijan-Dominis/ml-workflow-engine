"""Callbacks for ML scripts dashboard."""

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State

from ml_service.frontend.scripts.layout import PAGE_PREFIX
from ml_service.frontend.scripts.scripts_metadata import FRONTEND_SCRIPTS
from ml_service.frontend.scripts.utils import call_script


def register_callbacks(app):
    """Registers all script callbacks on the given Dash app."""

    def create_script_callbacks(script):
        """Creates callbacks for a given script to handle modal toggling and script execution."""
        input_ids = [f"{PAGE_PREFIX}-{script['name']}-{f['name']}" for f in script["fields"]]
        submit_id = f"{PAGE_PREFIX}-{script['name']}-submit"
        output_id = f"{PAGE_PREFIX}-{script['name']}-output"
        modal_id = f"{PAGE_PREFIX}-{script['name']}-modal"
        confirm_id = f"{PAGE_PREFIX}-{script['name']}-confirm"
        cancel_id = f"{PAGE_PREFIX}-{script['name']}-cancel"

        @app.callback(
            Output(modal_id, "is_open"),
            Input(submit_id, "n_clicks"),
            Input(confirm_id, "n_clicks"),
            Input(cancel_id, "n_clicks"),
            State(modal_id, "is_open"),
            prevent_initial_call=True
        )
        def toggle_modal(submit_click, confirm_click, cancel_click, is_open):
            """Toggles the confirmation modal based on which button was clicked."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return is_open
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == submit_id:
                return True
            elif button_id in [cancel_id, confirm_id]:
                return False
            return is_open

        @app.callback(
            Output(output_id, "children"),
            Input(confirm_id, "n_clicks"),
            [State(iid, "value") for iid in input_ids],
            prevent_initial_call=True
        )
        def run_pipeline(n_clicks, *values):
            """Executes the pipeline when the confirm button is clicked and displays the result."""
            if n_clicks is None:
                return dash.no_update

            payload = {}
            field_names = [f["name"] for f in script["fields"]]

            for k, v, f in zip(field_names, values, script["fields"], strict=True):
                if f["name"] == "operators" and isinstance(v, str):
                    payload[k] = [x.strip() for x in v.split(",")] if v else []

                elif f["type"] == "boolean":
                    payload[k] = bool(v) if v is not None else False

                elif f["type"] == "number":
                    if v in [None, '']:
                        payload[k] = None
                    elif '.' in str(v):
                        payload[k] = float(v)
                    else:
                        payload[k] = int(v)

                elif f["type"] == "text" or f["type"] == "dropdown":
                    payload[k] = v if v not in [None, ''] else None

            print("Payload sent:", payload)
            result = call_script(script["endpoint"], payload)

            return dbc.Textarea(
                value=str(result),
                style={"width": "100%", "height": "200px"},
                className="mt-2",
                id=f"{PAGE_PREFIX}-{script['name']}-result",
                name=f"{PAGE_PREFIX}-{script['name']}-result"
            )

    for s in FRONTEND_SCRIPTS:
        create_script_callbacks(s)
