"""
app.py  –  Interactive Dash web dashboard for crime severity monitoring.

Run locally
-----------
    python app.py

Deploy on Render
----------------
    Start command : gunicorn app:server
    Build command : pip install -r requirements.txt
    Environment   : MODEL_PATH=models/best_model.pt  (optional override)
"""

import os
import random
from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer

# ── Project root & model path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_PATH    = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = BASE_PATH / "models" / "best_model.pt"

# ── Lazy imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.models import MultimodalFusionModel

# ── Constants ────────────────────────────────────────────────────────────────
SEVERITY_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# Google Maps style: Green → Orange → Red
SEVERITY_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
MAP_COLORS      = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}

AUDIO_CLASSES = [
    "gun_shot", "siren", "drilling", "car_horn", "dog_bark",
    "jackhammer", "engine_idling", "street_music", "children_playing", "air_conditioner",
]

VIZAG_LOCATIONS = [
    {"name": "Downtown Visakhapatnam", "lat": 17.6868, "lon": 83.2185},
    {"name": "Industrial Zone",        "lat": 17.7333, "lon": 83.3167},
    {"name": "Residential Block",      "lat": 17.6667, "lon": 83.2000},
    {"name": "Beach Road",             "lat": 17.7200, "lon": 83.3300},
    {"name": "City Center",            "lat": 17.6900, "lon": 83.2200},
    {"name": "MVP Colony",             "lat": 17.7400, "lon": 83.3100},
    {"name": "Gajuwaka",               "lat": 17.6500, "lon": 83.2200},
    {"name": "Seethammadhara",         "lat": 17.7260, "lon": 83.3100},
    {"name": "Rushikonda",             "lat": 17.7800, "lon": 83.3800},
    {"name": "Madhurawada",            "lat": 17.8100, "lon": 83.3600},
]


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    device    = torch.device("cpu")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model     = MultimodalFusionModel().to(device)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print(f"[App] Loaded model from {MODEL_PATH}")
    else:
        print(f"[App] WARNING – checkpoint not found. Using demo data.")
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_precautionary_measures(severity: int, audio_class: str = "Unknown") -> list:
    base = {
        0: ["Log incident", "Continue monitoring", "Routine check"],
        1: ["Increase surveillance", "Alert security team", "Notify management"],
        2: ["Immediate lockdown", "Alert police/emergency", "Evacuate if safe", "Activate full protocol"],
    }
    measures = list(base.get(severity, []))
    if audio_class not in ("Unknown", ""):
        measures.append(f"Audio trigger: {audio_class}")
    return measures


def incidents_to_df(incidents: list) -> pd.DataFrame:
    df = pd.DataFrame(incidents)
    if df.empty:
        return df
    df["timestamp"]      = pd.to_datetime(df["timestamp"])
    df["severity_label"] = df["severity"].map(SEVERITY_LABELS)
    df["color"]          = df["severity"].map(SEVERITY_COLORS)
    df["map_color"]      = df["severity"].map(MAP_COLORS)
    return df


# ── Demo incidents ────────────────────────────────────────────────────────────
DEMO_INCIDENTS = [
    {
        "id":            f"INC-{i:03d}",
        "location":      VIZAG_LOCATIONS[i % len(VIZAG_LOCATIONS)]["name"],
        "lat":           VIZAG_LOCATIONS[i % len(VIZAG_LOCATIONS)]["lat"],
        "lon":           VIZAG_LOCATIONS[i % len(VIZAG_LOCATIONS)]["lon"],
        "timestamp":     (datetime.now() - pd.Timedelta(minutes=i * 15)).strftime("%Y-%m-%d %H:%M:%S"),
        "audio_class":   AUDIO_CLASSES[i % len(AUDIO_CLASSES)],
        "text_category": "CRIME",
        "text_content":  f"Demo incident {i} – replace with real test_loader data.",
        "severity":      i % 3,
        "confidence":    round(0.75 + 0.05 * (i % 5), 2),
        "measures":      get_precautionary_measures(i % 3, AUDIO_CLASSES[i % len(AUDIO_CLASSES)]),
    }
    for i in range(10)
]

df = incidents_to_df(DEMO_INCIDENTS)


# ── Shared axis style for dark background charts ──────────────────────────────
DARK_AXIS = dict(
    showgrid=True,
    gridcolor="#444444",
    gridwidth=1,
    zeroline=True,
    zerolinecolor="#666666",
    linecolor="#888888",
    linewidth=1,
    tickfont=dict(color="#e0e0e0"),
    title_font=dict(color="#e0e0e0"),
)


# ── Figure factories ──────────────────────────────────────────────────────────
def create_severity_chart(data_df: pd.DataFrame):
    counts = (
        data_df["severity_label"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .reset_index()
    )
    counts.columns = ["Severity", "Count"]
    fig = px.bar(
        counts, x="Severity", y="Count",
        color="Severity",
        color_discrete_map={"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"},
        text="Count",
    )
    fig.update_layout(
        title=dict(text="Incident Severity Distribution", font=dict(color="#e0e0e0")),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
        xaxis=dict(**DARK_AXIS, title="Severity Level"),
        yaxis=dict(**DARK_AXIS, title="Number of Incidents"),
    )
    fig.update_traces(textfont_color="#ffffff", textposition="outside")
    return fig


def create_map(data_df: pd.DataFrame):
    if data_df.empty:
        return px.scatter_mapbox(
            zoom=10, center={"lat": 17.6868, "lon": 83.2185},
            mapbox_style="open-street-map",
        )
    fig = px.scatter_mapbox(
        data_df,
        lat="lat", lon="lon",
        color="severity_label",
        color_discrete_map={"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"},
        size=[16] * len(data_df),
        hover_name="id",
        hover_data=["location", "timestamp", "audio_class", "text_content", "confidence"],
        zoom=10,
        center={"lat": 17.6868, "lon": 83.2185},
        mapbox_style="open-street-map",
    )
    fig.update_layout(
        title=dict(text="Geographic Distribution of Incidents", font=dict(color="#333333")),
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        legend_title="Severity",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#333333"},
    )
    fig.update_traces(marker=dict(opacity=0.85))
    return fig


def create_timeseries(data_df: pd.DataFrame):
    if data_df.empty:
        return px.line(title="No data available")
    df2 = data_df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    df2["hour"]      = df2["timestamp"].dt.floor("h")
    counts = df2.groupby(["hour", "severity_label"]).size().reset_index(name="count")
    fig = px.line(
        counts, x="hour", y="count",
        color="severity_label",
        color_discrete_map={"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"},
        markers=True,
        title="Severity Trends Over Time",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
        title_font=dict(color="#e0e0e0"),
        xaxis=dict(**DARK_AXIS, title="Time"),
        yaxis=dict(**DARK_AXIS, title="Number of Incidents"),
        legend_title="Severity",
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Crime Severity Monitor",
)
server = app.server  # for gunicorn

app.layout = dbc.Container([

    # ── Header ──────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.Div([
        html.H1("Multimodal Crime Severity Dashboard",
                className="text-center my-4", style={"color": "#17a2b8"}),
        html.P("Real-time detection using audio & text modalities | Locations are simulated for demo purposes",
               className="text-center text-muted mb-4"),
    ]))),

    # ── Confidence Threshold Slider ──────────────────────────────────────────
    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
        html.Label("🎚️ Confidence Threshold Filter",
                   style={"color": "#e0e0e0", "fontWeight": "bold", "fontSize": "1.1em"}),
        dcc.Slider(
            id="confidence-slider",
            min=0.0, max=1.0, step=0.05,
            value=0.5,
            marks={0: "0%", 0.25: "25%", 0.5: "50%", 0.75: "75%", 1.0: "100%"},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.P("Only show incidents at or above this confidence level",
               style={"color": "#adb5bd", "fontSize": "0.85em", "marginTop": "8px"}),
    ]), className="shadow mb-4"))),

    # ── Charts row ───────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(id="severity-chart")
        ), className="shadow"), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(id="incident-map")
        ), className="shadow"), width=6),
    ], className="mb-4"),

    # ── Time-series chart ────────────────────────────────────────────────────
    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody(
        dcc.Graph(id="timeseries-chart")
    ), className="shadow mb-4"))),

    # ── High severity alerts ─────────────────────────────────────────────────
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("⚠ High Severity Alerts", className="bg-danger text-white fw-bold"),
        dbc.CardBody(id="high-severity-list"),
    ], className="shadow mb-4"))),

    # ── Incidents table ──────────────────────────────────────────────────────
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("All Incidents", className="bg-primary text-white fw-bold"),
        dbc.CardBody(id="incidents-table"),
    ], className="shadow"))),

    # ── Stores & intervals ───────────────────────────────────────────────────
    dcc.Store(id="stored-incidents", data=DEMO_INCIDENTS),
    dcc.Store(id="filtered-incidents", data=DEMO_INCIDENTS),
    dcc.Interval(id="interval-component", interval=30_000, n_intervals=0),

    # ── Modal ────────────────────────────────────────────────────────────────
    dbc.Modal([
        dbc.ModalHeader(html.H4(id="modal-title"),
                        style={"background-color": "#2c3e50", "color": "#e0e0e0"}),
        dbc.ModalBody(id="modal-content",
                      style={"background-color": "#212529", "color": "#e0e0e0"}),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto", color="secondary")
        ),
    ], id="incident-modal", size="lg"),

], fluid=True, style={"background-color": "#121212", "min-height": "100vh",
                      "padding": "20px", "color": "#e0e0e0"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("filtered-incidents", "data"),
    Input("confidence-slider", "value"),
    State("stored-incidents", "data"),
)
def filter_by_confidence(threshold, incidents):
    return [inc for inc in incidents if inc["confidence"] >= threshold]


@callback(Output("severity-chart", "figure"), Input("filtered-incidents", "data"))
def update_severity_chart(incidents):
    return create_severity_chart(incidents_to_df(incidents))


@callback(Output("incident-map", "figure"), Input("filtered-incidents", "data"))
def update_map(incidents):
    return create_map(incidents_to_df(incidents))


@callback(Output("timeseries-chart", "figure"), Input("filtered-incidents", "data"))
def update_timeseries(incidents):
    return create_timeseries(incidents_to_df(incidents))


@callback(Output("high-severity-list", "children"), Input("filtered-incidents", "data"))
def update_high_severity(incidents):
    high = sorted(
        [inc for inc in incidents if inc["severity"] == 2],
        key=lambda x: x["confidence"], reverse=True,
    )
    if not high:
        return html.P("No high severity incidents above threshold.",
                      className="text-center p-3", style={"color": "#e0e0e0"})
    cards = []
    for inc in high:
        cards.append(dbc.Card(dbc.CardBody([
            html.H5(f"ID: {inc['id']} | Location: {inc['location']}",
                    style={"color": "#e0e0e0"}),
            html.P(f"Time: {inc['timestamp']} | Audio: {inc['audio_class']}",
                   style={"color": "#adb5bd"}),
            html.P(inc["text_content"], style={"color": "#e0e0e0"}),
            html.Div([
                html.P("Recommended Measures:", className="mb-1",
                       style={"color": "#ffc107", "fontWeight": "bold"}),
                html.Ul([html.Li(m, style={"color": "#e0e0e0"}) for m in inc["measures"][:4]]),
            ]),
            dbc.Button("View Details",
                       id={"type": "high-button", "index": inc["id"]},
                       color="danger", className="mt-2"),
        ], style={"background-color": "#212529"}), className="mb-3 shadow border-0"))
    return cards


@callback(Output("incidents-table", "children"), Input("filtered-incidents", "data"))
def update_table(incidents):
    sorted_inc = sorted(incidents, key=lambda x: x["timestamp"], reverse=True)
    header = html.Thead(html.Tr([
        html.Th(col, style={"color": "#ffffff", "fontWeight": "bold"})
        for col in ["ID", "Location", "Time", "Audio", "Text", "Severity", "Confidence", "Action"]
    ], style={"background-color": "#1a1a2e"}))

    row_bg    = {0: "#1a2b1a", 1: "#2b2415", 2: "#2b1515"}
    sev_badge = {0: "success", 1: "warning", 2: "danger"}

    rows = [
        html.Tr([
            html.Td(inc["id"],            style={"color": "#f0f0f0", "fontWeight": "500"}),
            html.Td(inc["location"],      style={"color": "#f0f0f0"}),
            html.Td(inc["timestamp"],     style={"color": "#d0d0d0", "fontSize": "0.9em"}),
            html.Td(inc["audio_class"],   style={"color": "#c0c0c0"}),
            html.Td(inc["text_category"], style={"color": "#c0c0c0"}),
            html.Td(html.Span(SEVERITY_LABELS[inc["severity"]],
                              className=f"badge bg-{sev_badge[inc['severity']]}",
                              style={"fontSize": "0.9em"})),
            html.Td(f"{inc['confidence']:.0%}", style={"color": "#e0e0e0"}),
            html.Td(dbc.Button("View Details",
                               id={"type": "table-button", "index": inc["id"]},
                               color="info", size="sm", outline=True)),
        ], style={"background-color": row_bg[inc["severity"]], "borderBottom": "1px solid #333"})
        for inc in sorted_inc
    ]
    return dbc.Table([header, html.Tbody(rows)],
                     bordered=True, hover=True, responsive=True,
                     style={"borderColor": "#333"})


def _modal_content(inc: dict):
    return html.Div([
        html.H5("Basic Information", style={"color": "#17a2b8"}),
        dbc.Row([
            dbc.Col(html.P(f"Location: {inc['location']}")),
            dbc.Col(html.P(f"Time: {inc['timestamp']}")),
        ]),
        dbc.Row([
            dbc.Col(html.P(f"Audio Class: {inc['audio_class']}")),
            dbc.Col(html.P(f"Text Category: {inc['text_category']}")),
        ]),
        html.H5("Description", style={"color": "#17a2b8"}),
        html.P(inc["text_content"]),
        html.H5("Severity Assessment", style={"color": "#17a2b8"}),
        dbc.Progress(
            value=inc["confidence"] * 100,
            color=["success", "warning", "danger"][inc["severity"]],
            striped=True, animated=True, style={"height": "25px"},
        ),
        html.P(f"Severity: {SEVERITY_LABELS[inc['severity']]} "
               f"(Confidence: {inc['confidence']:.1%})"),
        html.H5("Recommended Measures", style={"color": "#17a2b8"}),
        html.Ul([html.Li(m) for m in inc["measures"]]),
        html.H5("Location Map", style={"color": "#17a2b8"}),
        dcc.Graph(figure=px.scatter_mapbox(
            pd.DataFrame([inc]), lat="lat", lon="lon",
            zoom=14, mapbox_style="open-street-map",
        )),
    ], style={"color": "#f0f0f0"})


@callback(
    [Output("incident-modal", "is_open", allow_duplicate=True),
     Output("modal-title",    "children", allow_duplicate=True),
     Output("modal-content",  "children", allow_duplicate=True)],
    Input({"type": "high-button", "index": dash.ALL}, "n_clicks"),
    State("stored-incidents", "data"),
    State("incident-modal",   "is_open"),
    prevent_initial_call=True,
)
def open_modal_high(n_clicks, incidents, is_open):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks):
        return dash.no_update, dash.no_update, dash.no_update
    inc_id = eval(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    inc    = next((i for i in incidents if i["id"] == inc_id), None)
    if inc:
        return True, f"Incident Details: {inc['id']}", _modal_content(inc)
    return is_open, dash.no_update, dash.no_update


@callback(
    [Output("incident-modal", "is_open"),
     Output("modal-title",    "children"),
     Output("modal-content",  "children")],
    Input({"type": "table-button", "index": dash.ALL}, "n_clicks"),
    State("stored-incidents", "data"),
    State("incident-modal",   "is_open"),
    prevent_initial_call=True,
)
def open_modal_table(n_clicks, incidents, is_open):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks):
        return dash.no_update, dash.no_update, dash.no_update
    inc_id = eval(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    inc    = next((i for i in incidents if i["id"] == inc_id), None)
    if inc:
        return True, f"Incident Details: {inc['id']}", _modal_content(inc)
    return is_open, dash.no_update, dash.no_update


@callback(
    Output("incident-modal", "is_open", allow_duplicate=True),
    Input("close-modal", "n_clicks"),
    State("incident-modal", "is_open"),
    prevent_initial_call=True,
)
def close_modal(n_clicks, is_open):
    return not is_open if n_clicks else is_open


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))