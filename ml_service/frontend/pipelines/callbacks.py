"""Callbacks for ML pipelines dashboard."""

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State

from ml_service.frontend.pipelines.layout import PAGE_PREFIX
from ml_service.frontend.pipelines.pipelines_metadata import FRONTEND_PIPELINES
from ml_service.frontend.pipelines.utils import call_pipeline


def register_callbacks(app):
    """Registers all pipeline callbacks on the given Dash app."""

    def create_pipeline_callbacks(pipeline):
        """Creates callbacks for a given pipeline to handle modal toggling and pipeline execution."""
        input_ids = [f"{PAGE_PREFIX}-{pipeline['name']}-{f['name']}" for f in pipeline["fields"]]
        submit_id = f"{PAGE_PREFIX}-{pipeline['name']}-submit"
        output_id = f"{PAGE_PREFIX}-{pipeline['name']}-output"
        modal_id = f"{PAGE_PREFIX}-{pipeline['name']}-modal"
        confirm_id = f"{PAGE_PREFIX}-{pipeline['name']}-confirm"
        cancel_id = f"{PAGE_PREFIX}-{pipeline['name']}-cancel"

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
            field_names = [f["name"] for f in pipeline["fields"]]

            for k, v, f in zip(field_names, values, pipeline["fields"], strict=True):
                if f["type"] == "boolean":
                    payload[k] = bool(v) if v is not None else False
                elif f["type"] == "number":
                    payload[k] = int(v) if v not in [None, ''] else None
                elif f["type"] == "text" or f["type"] == "dropdown":
                    payload[k] = v if v not in [None, ''] else None

            print("Payload sent:", payload)
            result = call_pipeline(pipeline["endpoint"], payload)

            return dbc.Textarea(
                value=str(result),
                style={"width": "100%", "height": "200px"},
                className="mt-2",
                id=f"{PAGE_PREFIX}-{pipeline['name']}-result",
                name=f"{PAGE_PREFIX}-{pipeline['name']}-result"
            )

    for p in FRONTEND_PIPELINES:
        create_pipeline_callbacks(p)
