import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from pipelines_metadata import FRONTEND_PIPELINES

from utils import call_pipeline

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

tabs = []
for pipeline in FRONTEND_PIPELINES:
    inputs = []
    for field in pipeline["fields"]:
        input_id = f"{pipeline['name']}-{field['name']}"
        field_type = field["type"]
        field_value = field.get("value")

        if field_type == "text":
            inputs.append(dbc.Input(id=input_id, placeholder=field.get("placeholder", ""), type="text", value=field_value or ""))
        elif field_type == "number":
            inputs.append(dbc.Input(id=input_id, placeholder=field.get("placeholder", ""), type="number", value=field_value))
        elif field_type == "boolean":
            inputs.append(dbc.Checklist(
                options=[{"label": field["name"], "value": True}],
                value=[field_value] if field_value else [],
                id=input_id,
                inline=True
            ))
        elif field_type == "dropdown":
            default_value = field_value if field_value is not None else (field["options"][0] if field.get("options") else None)
            inputs.append(dcc.Dropdown(
                id=input_id,
                options=[{"label": opt, "value": opt} for opt in field.get("options", [])],
                value=default_value
            ))
    inputs.append(dbc.Button("Run Pipeline", id=f"{pipeline['name']}-submit", color="primary", className="mt-2"))
    inputs.append(html.Div(id=f"{pipeline['name']}-output", className="mt-2"))
    tabs.append(dbc.Tab(dbc.Card(dbc.CardBody(inputs)), label=pipeline["name"]))

app.layout = dbc.Container([
    html.H1("ML Pipelines Dashboard"),
    dbc.Tabs(tabs)
], fluid=True)

for pipeline in FRONTEND_PIPELINES:
    input_ids = [f"{pipeline['name']}-{field['name']}" for field in pipeline["fields"]]
    submit_id = f"{pipeline['name']}-submit"
    output_id = f"{pipeline['name']}-output"

    @app.callback(
        Output(output_id, "children"),
        Input(submit_id, "n_clicks"),
        [State(input_id, "value") for input_id in input_ids],
        prevent_initial_call=True
    )
    def run_pipeline(n_clicks, *values, pipeline_name=pipeline["name"], field_names=[f["name"] for f in pipeline["fields"]], field_defs=pipeline["fields"]):
        payload = {}
        for k, v, f in zip(field_names, values, field_defs):
            if f["type"] == "boolean":
                payload[k] = True if v else False
            elif f["type"] in ("text", "dropdown"):
                payload[k] = v if v not in ("", None) else None
            else:
                payload[k] = v
        result = call_pipeline(pipeline_name, payload)
        return dbc.Textarea(value=str(result), style={"width": "100%", "height": "200px"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
