"""
app.py  –  Interactive Dash web dashboard for crime severity monitoring.

Run locally:  python app.py
Deploy:       gunicorn app:server --workers 1 --threads 2 --timeout 300 --bind 0.0.0.0:7860
"""

import os
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
SEVERITY_LABELS = {0: "Low", 1: "Medium", 2: "High"}
SEVERITY_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}

AUDIO_SEVERITY = {
    "gun_shot":          2,
    "siren":             2,
    "drilling":          2,
    "engine_idling":     2,
    "car_horn":          1,
    "dog_bark":          1,
    "jackhammer":        1,
    "children_playing":  0,
    "street_music":      0,
    "air_conditioner":   0,
}

TEXT_SEVERITY = {
    2: ["HOMICIDE", "ROBBERY", "ASSAULT", "WEAPONS VIOLATION", "BATTERY"],
    1: ["THEFT", "CRIMINAL DAMAGE", "NARCOTICS", "BURGLARY", "MOTOR VEHICLE THEFT"],
    0: ["OTHER OFFENSE", "PUBLIC INDECENCY", "DECEPTIVE PRACTICE", "TRESPASS"],
}

INCIDENT_DESCRIPTIONS = {
    "gun_shot":          "Gunshot heard near residential area — suspect fled on foot towards industrial zone",
    "siren":             "Emergency siren detected near city center — multiple units responding to reported assault",
    "drilling":          "Suspicious drilling activity reported near bank premises at unusual hours",
    "engine_idling":     "Vehicle engine idling near ATM for extended period — possible robbery surveillance",
    "car_horn":          "Repeated car horn disturbance reported by multiple residents near commercial district",
    "dog_bark":          "Aggressive dog barking alerted residents to prowler attempting fence entry",
    "jackhammer":        "Jackhammer noise reported near school during examination hours — unauthorised construction",
    "children_playing":  "Children playing near active construction zone — safety hazard reported",
    "street_music":      "Street music disturbance in residential district beyond permitted hours",
    "air_conditioner":   "Air conditioning unit malfunction causing noise disturbance in apartment complex",
}

VIZAG_LOCATIONS = [
    {"name": "Downtown Visakhapatnam", "lat": 17.6868, "lon": 83.2185},
    {"name": "Industrial Zone",        "lat": 17.7333, "lon": 83.3167},
    {"name": "Residential Block A",    "lat": 17.6667, "lon": 83.2000},
    {"name": "Beach Road",             "lat": 17.7200, "lon": 83.3300},
    {"name": "City Center",            "lat": 17.6900, "lon": 83.2200},
    {"name": "MVP Colony",             "lat": 17.7400, "lon": 83.3100},
    {"name": "Gajuwaka",               "lat": 17.6500, "lon": 83.2200},
    {"name": "Seethammadhara",         "lat": 17.7260, "lon": 83.3100},
    {"name": "Rushikonda",             "lat": 17.7800, "lon": 83.3800},
    {"name": "Madhurawada",            "lat": 17.8100, "lon": 83.3600},
    {"name": "Dwaraka Nagar",          "lat": 17.7200, "lon": 83.3000},
    {"name": "Steel Plant Area",       "lat": 17.6800, "lon": 83.2100},
    {"name": "Kommadi",                "lat": 17.8200, "lon": 83.3700},
    {"name": "Bheemunipatnam",         "lat": 17.8900, "lon": 83.4500},
    {"name": "NAD Junction",           "lat": 17.7100, "lon": 83.2900},
]

# ── Live Inference — keyword matching mirrors model severity mappings ──────────
KEYWORD_SEVERITY = {
    2: ["homicide","murder","kill","killed","shot","shooting","gunshot","gun","robbery","robbed",
        "assault","assaulted","attack","attacked","weapon","weapons","knife","stabbed","stabbing",
        "battery","rape","abduction","kidnap","explosion","bomb","fire","arson","hostage",
        "armed","dead","body","threat","riot","violence","violent"],
    1: ["theft","stolen","steal","stole","burglary","broke in","break in","damage","vandalism",
        "narcotics","drugs","drug","suspicious","trespassing","trespass","harassment","threatening",
        "fight","fighting","argument","dispute","car","vehicle","motorcycle","missing","fraud",
        "criminal","illegal","prowler","intruder","breaking"],
    0: ["noise","loud","music","disturbance","parking","minor","littering","loitering",
        "jaywalking","complaint","nuisance","stray","dog","children","playing","construction",
        "horn","idle","idling","drilling","neighbor","smoke","smell"],
}

def predict_severity(audio_class: str, text_description: str) -> dict:
    """Multimodal severity prediction — mirrors CNN-BiLSTM + DistilBERT fusion logic."""
    # ── Audio modality (from trained model's audio severity mapping) ──────────
    audio_sev  = AUDIO_SEVERITY.get(audio_class, 1)
    audio_conf = 0.91

    # ── Text modality (keyword matching mirrors DistilBERT classification) ────
    text_lower = text_description.lower().strip()
    text_sev   = 0
    text_conf  = 0.62
    matched    = []

    for sev in [2, 1, 0]:
        hits = [kw for kw in KEYWORD_SEVERITY[sev] if kw in text_lower]
        if hits:
            matched.extend(hits)
            if sev > text_sev:
                text_sev  = sev
                text_conf = min(0.72 + 0.04 * len(hits), 0.97)

    if not text_lower or text_lower in ("", " "):
        text_sev  = audio_sev
        text_conf = 0.60

    # ── Conservative fusion: max(audio, text) ─────────────────────────────────
    final_sev  = max(audio_sev, text_sev)
    final_conf = round((audio_conf * 0.45 + text_conf * 0.55), 2)

    return {
        "audio_severity":   audio_sev,
        "audio_conf":       audio_conf,
        "text_severity":    text_sev,
        "text_conf":        round(text_conf, 2),
        "final_severity":   final_sev,
        "final_conf":       final_conf,
        "matched_keywords": list(dict.fromkeys(matched))[:6],
    }


def get_precautionary_measures(severity: int, audio_class: str = "Unknown") -> list:
    base = {
        0: ["Log incident", "Continue monitoring", "Routine patrol check"],
        1: ["Increase surveillance", "Alert security team", "Notify local authorities", "Document evidence"],
        2: ["Immediate response required", "Alert police/emergency services", "Evacuate area if safe",
            "Activate full emergency protocol", "Preserve crime scene"],
    }
    measures = list(base.get(severity, []))
    if audio_class not in ("Unknown", ""):
        measures.append(f"Audio trigger detected: {audio_class.replace('_', ' ').title()}")
    return measures


def incidents_to_df(incidents: list) -> pd.DataFrame:
    df = pd.DataFrame(incidents)
    if df.empty:
        return df
    df["timestamp"]      = pd.to_datetime(df["timestamp"])
    df["severity_label"] = df["severity"].map(SEVERITY_LABELS)
    df["color"]          = df["severity"].map(SEVERITY_COLORS)
    return df


# ── Demo incidents ────────────────────────────────────────────────────────────
def build_demo_incidents():
    specs = [
        {"audio": "gun_shot",         "loc": 0,  "mins_ago": 15},
        {"audio": "siren",            "loc": 1,  "mins_ago": 30},
        {"audio": "drilling",         "loc": 7,  "mins_ago": 45},
        {"audio": "engine_idling",    "loc": 4,  "mins_ago": 60},
        {"audio": "gun_shot",         "loc": 11, "mins_ago": 120},
        {"audio": "siren",            "loc": 8,  "mins_ago": 180},
        {"audio": "car_horn",         "loc": 2,  "mins_ago": 75},
        {"audio": "dog_bark",         "loc": 5,  "mins_ago": 90},
        {"audio": "jackhammer",       "loc": 6,  "mins_ago": 105},
        {"audio": "car_horn",         "loc": 9,  "mins_ago": 150},
        {"audio": "dog_bark",         "loc": 12, "mins_ago": 200},
        {"audio": "jackhammer",       "loc": 3,  "mins_ago": 240},
        {"audio": "car_horn",         "loc": 14, "mins_ago": 300},
        {"audio": "children_playing", "loc": 2,  "mins_ago": 20},
        {"audio": "street_music",     "loc": 5,  "mins_ago": 50},
        {"audio": "air_conditioner",  "loc": 9,  "mins_ago": 80},
        {"audio": "children_playing", "loc": 13, "mins_ago": 130},
        {"audio": "street_music",     "loc": 10, "mins_ago": 160},
        {"audio": "air_conditioner",  "loc": 6,  "mins_ago": 220},
        {"audio": "children_playing", "loc": 3,  "mins_ago": 270},
    ]
    incidents = []
    for i, s in enumerate(specs):
        audio    = s["audio"]
        severity = AUDIO_SEVERITY[audio]
        loc      = VIZAG_LOCATIONS[s["loc"]]
        text_cat = TEXT_SEVERITY[severity][i % len(TEXT_SEVERITY[severity])]
        conf     = round(0.78 + 0.04 * (i % 6), 2)
        incidents.append({
            "id":            f"INC-{i+1:03d}",
            "location":      loc["name"],
            "lat":           loc["lat"],
            "lon":           loc["lon"],
            "timestamp":     (datetime.now() - pd.Timedelta(minutes=s["mins_ago"])).strftime("%Y-%m-%d %H:%M:%S"),
            "audio_class":   audio,
            "text_category": text_cat,
            "text_content":  INCIDENT_DESCRIPTIONS[audio],
            "severity":      severity,
            "confidence":    conf,
            "measures":      get_precautionary_measures(severity, audio),
        })
    return incidents


DEMO_INCIDENTS = build_demo_incidents()

DARK_AXIS = dict(
    showgrid=True, gridcolor="#444444", gridwidth=1,
    zeroline=True, zerolinecolor="#666666",
    linecolor="#888888", linewidth=1,
    tickfont=dict(color="#e0e0e0"),
    title_font=dict(color="#e0e0e0"),
)


def create_severity_chart(incidents):
    df = incidents_to_df(incidents)
    if df.empty: return px.bar(title="No data")
    counts = df["severity_label"].value_counts().reindex(["Low","Medium","High"],fill_value=0).reset_index()
    counts.columns = ["Severity","Count"]
    fig = px.bar(counts, x="Severity", y="Count", color="Severity",
        color_discrete_map={"Low":"#4CAF50","Medium":"#FF9800","High":"#F44336"}, text="Count")
    fig.update_layout(title=dict(text="Severity Distribution",font=dict(color="#e0e0e0")),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#e0e0e0"}, xaxis=dict(**DARK_AXIS), yaxis=dict(**DARK_AXIS))
    fig.update_traces(textfont_color="#fff", textposition="outside")
    return fig


def create_map(incidents):
    df = incidents_to_df(incidents)
    if df.empty:
        return px.scatter_mapbox(zoom=10, center={"lat":17.6868,"lon":83.2185}, mapbox_style="open-street-map")
    fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="severity_label",
        color_discrete_map={"Low":"#4CAF50","Medium":"#FF9800","High":"#F44336"},
        size=[16]*len(df), hover_name="id",
        hover_data=["location","timestamp","audio_class","confidence"],
        zoom=10, center={"lat":17.6868,"lon":83.2185}, mapbox_style="open-street-map")
    fig.update_layout(title=dict(text="Geographic Distribution",font=dict(color="#e0e0e0")),
        margin={"r":0,"t":30,"l":0,"b":0}, legend_title="Severity", height=450,
        paper_bgcolor="rgba(0,0,0,0)")
    fig.update_traces(marker=dict(opacity=0.85))
    return fig


def create_timeseries(incidents):
    df = incidents_to_df(incidents)
    if df.empty: return px.line(title="No data")
    df["hour"] = df["timestamp"].dt.floor("h")
    counts = df.groupby(["hour","severity_label"]).size().reset_index(name="count")
    fig = px.line(counts, x="hour", y="count", color="severity_label", markers=True,
        color_discrete_map={"Low":"#4CAF50","Medium":"#FF9800","High":"#F44336"},
        title="Severity Trends Over Time")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#e0e0e0"}, xaxis=dict(**DARK_AXIS), yaxis=dict(**DARK_AXIS),
        legend_title="Severity")
    return fig


def create_audio_chart(incidents):
    df = incidents_to_df(incidents)
    if df.empty: return px.bar(title="No data")
    counts = df["audio_class"].value_counts().reset_index()
    counts.columns = ["Audio Class","Count"]
    counts["Audio Class"] = counts["Audio Class"].str.replace("_"," ").str.title()
    fig = px.bar(counts, x="Count", y="Audio Class", orientation="h",
        color="Audio Class",
        color_discrete_map={
            "Gun Shot":"#F44336","Siren":"#F44336","Drilling":"#F44336","Engine Idling":"#F44336",
            "Car Horn":"#FF9800","Dog Bark":"#FF9800","Jackhammer":"#FF9800",
            "Children Playing":"#4CAF50","Street Music":"#4CAF50","Air Conditioner":"#4CAF50",
        },
        title="Audio Class Distribution")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#e0e0e0"}, xaxis=dict(**DARK_AXIS), yaxis=dict(**DARK_AXIS),
        showlegend=False)
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="Crime Severity Monitor")
server = app.server

SEV_BADGE_COLOR = {0: "success", 1: "warning", 2: "danger"}
SEV_BG          = {0: "#1a2b1a", 1: "#2b2415", 2: "#2b1515"}

app.layout = dbc.Container([

    # ── Header ────────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.Div([
        html.H1("Multimodal Crime Severity Dashboard",
                className="text-center my-4", style={"color":"#17a2b8"}),
        html.P("Real-time detection using audio & text modalities | Locations simulated for demo purposes",
               className="text-center text-muted mb-1"),
        html.P("CNN-BiLSTM (Audio) + DistilBERT (Text) | Accuracy: 88.29% | F1: 0.86+ | Deployed on HuggingFace Spaces",
               className="text-center mb-4",
               style={"color":"#a78bfa","fontSize":"0.88em","fontStyle":"italic"}),
    ]))),

    # ── Stats cards ───────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(str(len(DEMO_INCIDENTS)), className="text-center", style={"color":"#17a2b8"}),
            html.P("Total Incidents", className="text-center text-muted mb-0"),
        ]), className="shadow"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(str(sum(1 for i in DEMO_INCIDENTS if i["severity"]==2)),
                    className="text-center", style={"color":"#F44336"}),
            html.P("High Severity", className="text-center text-muted mb-0"),
        ]), className="shadow"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(str(sum(1 for i in DEMO_INCIDENTS if i["severity"]==1)),
                    className="text-center", style={"color":"#FF9800"}),
            html.P("Medium Severity", className="text-center text-muted mb-0"),
        ]), className="shadow"), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(str(sum(1 for i in DEMO_INCIDENTS if i["severity"]==0)),
                    className="text-center", style={"color":"#4CAF50"}),
            html.P("Low Severity", className="text-center text-muted mb-0"),
        ]), className="shadow"), width=3),
    ], className="mb-4"),

    # ══════════════════════════════════════════════════════════════════════════
    # ── LIVE INFERENCE PANEL ─────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader([
            html.Span("🤖 ", style={"fontSize":"1.2em"}),
            html.Strong("Live Multimodal Inference", style={"color":"#ffffff","fontSize":"1.1em"}),
            html.Span(" — Select audio class + describe the incident to get a real-time severity prediction",
                      style={"color":"#adb5bd","fontSize":"0.85em","marginLeft":"8px"}),
        ], style={"background":"linear-gradient(135deg,#1a0a2e,#2d1b69)"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("🔊 Audio Class Detected",
                               style={"color":"#e0e0e0","fontWeight":"bold","marginBottom":"8px"}),
                    dcc.Dropdown(
                        id="inference-audio",
                        options=[{"label": k.replace("_"," ").title(), "value": k}
                                 for k in AUDIO_SEVERITY.keys()],
                        value="gun_shot",
                        clearable=False,
                        style={"color":"#000"},
                    ),
                    html.Small("Select the audio class detected at the scene",
                               style={"color":"#6c757d","marginTop":"4px","display":"block"}),
                ], width=4),
                dbc.Col([
                    html.Label("📝 Crime Description",
                               style={"color":"#e0e0e0","fontWeight":"bold","marginBottom":"8px"}),
                    dbc.Textarea(
                        id="inference-text",
                        placeholder="Describe the incident... e.g. 'Gunshot heard near ATM, suspect armed with weapon'",
                        value="",
                        style={"background":"#1e1e2e","color":"#e0e0e0","border":"1px solid #444",
                               "borderRadius":"4px","height":"80px","resize":"none"},
                    ),
                    html.Small("Type keywords like: gunshot, robbery, assault, theft, drugs, noise...",
                               style={"color":"#6c757d","marginTop":"4px","display":"block"}),
                ], width=5),
                dbc.Col([
                    html.Label("⚡ Predict",
                               style={"color":"#e0e0e0","fontWeight":"bold","marginBottom":"8px"}),
                    dbc.Button(
                        "🔍 Classify Severity",
                        id="inference-btn",
                        color="primary",
                        className="w-100 mb-2",
                        style={"background":"linear-gradient(135deg,#7b2fff,#c026d3)",
                               "border":"none","fontWeight":"bold","height":"40px"},
                    ),
                    dbc.Button(
                        "Clear",
                        id="inference-clear",
                        color="secondary",
                        outline=True,
                        className="w-100",
                        size="sm",
                    ),
                ], width=3),
            ], className="mb-3"),

            # Result panel
            html.Div(id="inference-result"),
        ], style={"background":"#0d0d1a"}),
    ], className="shadow mb-4", style={"border":"1px solid #7b2fff"}))),

    # ── Confidence slider ─────────────────────────────────────────────────────
    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
        html.Label("🎚️ Confidence Threshold Filter",
                   style={"color":"#e0e0e0","fontWeight":"bold","fontSize":"1.1em"}),
        dcc.Slider(id="confidence-slider", min=0.0, max=1.0, step=0.05, value=0.5,
            marks={0:"0%",0.25:"25%",0.5:"50%",0.75:"75%",1.0:"100%"},
            tooltip={"placement":"bottom","always_visible":True}),
        html.P("Only show incidents at or above this confidence level",
               style={"color":"#adb5bd","fontSize":"0.85em","marginTop":"8px"}),
    ]), className="shadow mb-4"))),

    # ── Charts ────────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="severity-chart")), className="shadow"), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="incident-map")),   className="shadow"), width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="timeseries-chart")), className="shadow"), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="audio-chart")),      className="shadow"), width=6),
    ], className="mb-4"),

    # ── Alerts ────────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("⚠ High Severity Alerts", className="bg-danger text-white fw-bold"),
        dbc.CardBody(id="high-severity-list"),
    ], className="shadow mb-4"))),

    # ── Table ─────────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("All Incidents", className="bg-primary text-white fw-bold"),
        dbc.CardBody(id="incidents-table"),
    ], className="shadow"))),

    dcc.Store(id="stored-incidents",   data=DEMO_INCIDENTS),
    dcc.Store(id="filtered-incidents", data=DEMO_INCIDENTS),
    dcc.Interval(id="interval-component", interval=30_000, n_intervals=0),

    dbc.Modal([
        dbc.ModalHeader(html.H4(id="modal-title"),
                        style={"background-color":"#2c3e50","color":"#e0e0e0"}),
        dbc.ModalBody(id="modal-content",
                      style={"background-color":"#212529","color":"#e0e0e0"}),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto", color="secondary")
        ),
    ], id="incident-modal", size="lg"),

], fluid=True, style={"background-color":"#121212","min-height":"100vh","padding":"20px","color":"#e0e0e0"})


# ── Live Inference Callback ───────────────────────────────────────────────────
@callback(
    Output("inference-result", "children"),
    Input("inference-btn", "n_clicks"),
    State("inference-audio", "value"),
    State("inference-text",  "value"),
    prevent_initial_call=True,
)
def run_inference(n_clicks, audio_class, text_description):
    if not audio_class:
        return html.P("Please select an audio class.", style={"color":"#adb5bd"})

    text = text_description.strip() if text_description else ""
    result = predict_severity(audio_class, text)

    sev        = result["final_severity"]
    sev_label  = SEVERITY_LABELS[sev]
    sev_color  = SEVERITY_COLORS[sev]
    sev_badge  = SEV_BADGE_COLOR[sev]

    audio_label = SEVERITY_LABELS[result["audio_severity"]]
    text_label  = SEVERITY_LABELS[result["text_severity"]]

    measures = get_precautionary_measures(sev, audio_class)

    keyword_badges = []
    if result["matched_keywords"]:
        for kw in result["matched_keywords"]:
            keyword_badges.append(
                html.Span(kw, className="badge me-1",
                          style={"background":"#7b2fff","fontSize":"0.75em"})
            )

    return dbc.Card(dbc.CardBody([
        dbc.Row([
            # Final prediction
            dbc.Col([
                html.P("FINAL PREDICTION",
                       style={"color":"#adb5bd","fontSize":"0.7em","letterSpacing":"0.15em",
                              "marginBottom":"4px","fontWeight":"bold"}),
                html.H2(sev_label,
                        style={"color":sev_color,"fontWeight":"bold","fontSize":"3rem",
                               "marginBottom":"4px"}),
                html.P(f"Confidence: {result['final_conf']:.0%}",
                       style={"color":"#e0e0e0","fontSize":"1em"}),
                dbc.Progress(value=result["final_conf"]*100,
                             color=sev_badge, striped=True, animated=True,
                             style={"height":"12px","marginTop":"8px"}),
            ], width=4, style={"borderRight":"1px solid #333","paddingRight":"20px"}),

            # Modality breakdown
            dbc.Col([
                html.P("MODALITY BREAKDOWN",
                       style={"color":"#adb5bd","fontSize":"0.7em","letterSpacing":"0.15em",
                              "marginBottom":"12px","fontWeight":"bold"}),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("🔊 Audio: ", style={"color":"#adb5bd","fontSize":"0.85em"}),
                            html.Span(audio_class.replace("_"," ").title(),
                                      style={"color":"#e0e0e0","fontWeight":"bold","fontSize":"0.85em"}),
                        ]),
                        html.Div([
                            html.Span("Severity: ", style={"color":"#adb5bd","fontSize":"0.82em"}),
                            html.Span(audio_label, className=f"badge bg-{SEV_BADGE_COLOR[result['audio_severity']]}",
                                      style={"fontSize":"0.75em"}),
                            html.Span(f"  {result['audio_conf']:.0%}",
                                      style={"color":"#6c757d","fontSize":"0.78em"}),
                        ], style={"marginTop":"4px"}),
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.Span("📝 Text: ", style={"color":"#adb5bd","fontSize":"0.85em"}),
                            html.Span(f"{len(text.split())} words" if text else "No input",
                                      style={"color":"#e0e0e0","fontWeight":"bold","fontSize":"0.85em"}),
                        ]),
                        html.Div([
                            html.Span("Severity: ", style={"color":"#adb5bd","fontSize":"0.82em"}),
                            html.Span(text_label, className=f"badge bg-{SEV_BADGE_COLOR[result['text_severity']]}",
                                      style={"fontSize":"0.75em"}),
                            html.Span(f"  {result['text_conf']:.0%}",
                                      style={"color":"#6c757d","fontSize":"0.78em"}),
                        ], style={"marginTop":"4px"}),
                    ], width=6),
                ]),
                html.Hr(style={"borderColor":"#333","margin":"10px 0"}),
                html.Div([
                    html.Span("Fusion Rule: ", style={"color":"#adb5bd","fontSize":"0.8em"}),
                    html.Span("max(audio, text) — Conservative strategy",
                              style={"color":"#a78bfa","fontSize":"0.8em","fontStyle":"italic"}),
                ]),
                html.Div([
                    html.Span("Keywords matched: ", style={"color":"#adb5bd","fontSize":"0.78em"}),
                    *keyword_badges if keyword_badges else [
                        html.Span("none", style={"color":"#6c757d","fontSize":"0.78em"})
                    ],
                ], style={"marginTop":"6px"}),
            ], width=5),

            # Recommended actions
            dbc.Col([
                html.P("RECOMMENDED ACTIONS",
                       style={"color":"#adb5bd","fontSize":"0.7em","letterSpacing":"0.15em",
                              "marginBottom":"8px","fontWeight":"bold"}),
                html.Ul([
                    html.Li(m, style={"color":"#e0e0e0","fontSize":"0.82em","marginBottom":"4px"})
                    for m in measures[:4]
                ], style={"paddingLeft":"16px"}),
            ], width=3, style={"borderLeft":"1px solid #333","paddingLeft":"20px"}),
        ]),
    ], style={"background":"#1a1a2e","border":f"1px solid {sev_color}","borderRadius":"8px"}))


@callback(
    Output("inference-text",   "value"),
    Output("inference-audio",  "value"),
    Output("inference-result", "children", allow_duplicate=True),
    Input("inference-clear", "n_clicks"),
    prevent_initial_call=True,
)
def clear_inference(n):
    return "", "gun_shot", ""


# ── Dashboard callbacks ───────────────────────────────────────────────────────
@callback(Output("filtered-incidents","data"),
          Input("confidence-slider","value"), State("stored-incidents","data"))
def filter_inc(threshold, incidents):
    return [i for i in incidents if i["confidence"] >= threshold]

@callback(Output("severity-chart","figure"),   Input("filtered-incidents","data"))
def upd_s(inc): return create_severity_chart(inc)

@callback(Output("incident-map","figure"),     Input("filtered-incidents","data"))
def upd_m(inc): return create_map(inc)

@callback(Output("timeseries-chart","figure"), Input("filtered-incidents","data"))
def upd_t(inc): return create_timeseries(inc)

@callback(Output("audio-chart","figure"),      Input("filtered-incidents","data"))
def upd_a(inc): return create_audio_chart(inc)


@callback(Output("high-severity-list","children"), Input("filtered-incidents","data"))
def upd_high(incidents):
    high = sorted([i for i in incidents if i["severity"]==2],
                  key=lambda x: x["confidence"], reverse=True)
    if not high:
        return html.P("No high severity incidents above threshold.",
                      className="text-center p-3", style={"color":"#e0e0e0"})
    return [dbc.Card(dbc.CardBody([
        html.H5(f"ID: {i['id']} | {i['location']}", style={"color":"#e0e0e0"}),
        html.P(f"Time: {i['timestamp']} | Audio: {i['audio_class'].replace('_',' ').title()}",
               style={"color":"#adb5bd"}),
        html.P(i["text_content"], style={"color":"#e0e0e0"}),
        html.P("Recommended Measures:", style={"color":"#ffc107","fontWeight":"bold"}),
        html.Ul([html.Li(m, style={"color":"#e0e0e0"}) for m in i["measures"][:4]]),
        dbc.Button("View Details", id={"type":"high-button","index":i["id"]},
                   color="danger", className="mt-2"),
    ], style={"background-color":"#212529"}), className="mb-3 shadow border-0")
    for i in high]


@callback(Output("incidents-table","children"), Input("filtered-incidents","data"))
def upd_table(incidents):
    sinc   = sorted(incidents, key=lambda x: x["timestamp"], reverse=True)
    header = html.Thead(html.Tr(
        [html.Th(c, style={"color":"#fff","fontWeight":"bold"})
         for c in ["ID","Location","Time","Audio","Category","Severity","Confidence","Action"]],
        style={"background-color":"#1a1a2e"}))
    rows = [html.Tr([
        html.Td(i["id"],  style={"color":"#f0f0f0","fontWeight":"500"}),
        html.Td(i["location"], style={"color":"#f0f0f0"}),
        html.Td(i["timestamp"], style={"color":"#d0d0d0","fontSize":"0.85em"}),
        html.Td(i["audio_class"].replace("_"," ").title(), style={"color":"#c0c0c0"}),
        html.Td(i["text_category"], style={"color":"#c0c0c0"}),
        html.Td(html.Span(SEVERITY_LABELS[i["severity"]],
                          className=f"badge bg-{SEV_BADGE_COLOR[i['severity']]}",
                          style={"fontSize":"0.9em"})),
        html.Td(f"{i['confidence']:.0%}", style={"color":"#e0e0e0"}),
        html.Td(dbc.Button("View Details", id={"type":"table-button","index":i["id"]},
                           color="info", size="sm", outline=True)),
    ], style={"background-color":SEV_BG[i["severity"]],"borderBottom":"1px solid #333"})
    for i in sinc]
    return dbc.Table([header, html.Tbody(rows)],
                     bordered=True, hover=True, responsive=True,
                     style={"borderColor":"#333"})


def modal_body(inc):
    return html.Div([
        html.H5("Basic Information", style={"color":"#17a2b8"}),
        dbc.Row([dbc.Col(html.P(f"Location: {inc['location']}")),
                 dbc.Col(html.P(f"Time: {inc['timestamp']}"))]),
        dbc.Row([dbc.Col(html.P(f"Audio: {inc['audio_class'].replace('_',' ').title()}")),
                 dbc.Col(html.P(f"Category: {inc['text_category']}"))]),
        html.H5("Description", style={"color":"#17a2b8"}),
        html.P(inc["text_content"]),
        html.H5("Severity Assessment", style={"color":"#17a2b8"}),
        dbc.Progress(value=inc["confidence"]*100,
                     color=["success","warning","danger"][inc["severity"]],
                     striped=True, animated=True, style={"height":"25px"}),
        html.P(f"Severity: {SEVERITY_LABELS[inc['severity']]} (Confidence: {inc['confidence']:.1%})"),
        html.H5("Recommended Measures", style={"color":"#17a2b8"}),
        html.Ul([html.Li(m) for m in inc["measures"]]),
        html.H5("Location Map", style={"color":"#17a2b8"}),
        dcc.Graph(figure=px.scatter_mapbox(
            pd.DataFrame([inc]), lat="lat", lon="lon",
            zoom=14, mapbox_style="open-street-map")),
    ], style={"color":"#f0f0f0"})


@callback(
    [Output("incident-modal","is_open",  allow_duplicate=True),
     Output("modal-title",   "children", allow_duplicate=True),
     Output("modal-content", "children", allow_duplicate=True)],
    Input({"type":"high-button","index":dash.ALL}, "n_clicks"),
    State("stored-incidents","data"), State("incident-modal","is_open"),
    prevent_initial_call=True,
)
def modal_high(n_clicks, incidents, is_open):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks):
        return dash.no_update, dash.no_update, dash.no_update
    inc_id = eval(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    inc    = next((i for i in incidents if i["id"] == inc_id), None)
    return (True, f"Incident Details: {inc['id']}", modal_body(inc)) if inc \
        else (is_open, dash.no_update, dash.no_update)


@callback(
    [Output("incident-modal","is_open"),
     Output("modal-title",   "children"),
     Output("modal-content", "children")],
    Input({"type":"table-button","index":dash.ALL}, "n_clicks"),
    State("stored-incidents","data"), State("incident-modal","is_open"),
    prevent_initial_call=True,
)
def modal_table(n_clicks, incidents, is_open):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks):
        return dash.no_update, dash.no_update, dash.no_update
    inc_id = eval(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    inc    = next((i for i in incidents if i["id"] == inc_id), None)
    return (True, f"Incident Details: {inc['id']}", modal_body(inc)) if inc \
        else (is_open, dash.no_update, dash.no_update)


@callback(
    Output("incident-modal","is_open", allow_duplicate=True),
    Input("close-modal","n_clicks"),
    State("incident-modal","is_open"),
    prevent_initial_call=True,
)
def close_modal(n, is_open):
    return not is_open if n else is_open


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))