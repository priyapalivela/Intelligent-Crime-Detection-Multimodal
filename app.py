"""
app.py  –  Interactive Dash web dashboard for crime severity monitoring.
v1.4.0 — Word attention visualization + improved color theme

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

# ── Severity colors — UNCHANGED ───────────────────────────────────────────────
SEVERITY_LABELS = {0: "Low", 1: "Medium", 2: "High"}
SEVERITY_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}

# ── Improved UI color palette ─────────────────────────────────────────────────
ACCENT       = "#7c3aed"
ACCENT2      = "#06b6d4"
ACCENT_GRAD  = "linear-gradient(135deg,#1a0a2e,#2d1b69)"
CARD_BG      = "#0f0f1a"
SURFACE      = "#16162a"
BORDER_COL   = "#2d2d4e"
TEXT_PRIMARY = "#f0f0f0"
TEXT_MUTED   = "#8b8fa8"
HEADER_GRAD  = "linear-gradient(135deg,#0f0720 0%,#1a0a2e 50%,#0d1117 100%)"

AUDIO_SEVERITY = {
    "gun_shot": 2, "siren": 2, "drilling": 2, "engine_idling": 2,
    "car_horn": 1, "dog_bark": 1, "jackhammer": 1,
    "children_playing": 0, "street_music": 0, "air_conditioner": 0,
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

KEYWORD_SEVERITY = {
    2: ["homicide","murder","kill","killed","shot","shooting","gunshot","gun","robbery","robbed",
        "assault","assaulted","attack","attacked","weapon","weapons","knife","stabbed","stabbing",
        "battery","rape","abduction","kidnap","kidnapping","kidnapped","explosion","bomb","fire",
        "arson","hostage","armed","dead","body","threat","riot","violence","violent","molest",
        "molested","trafficking","tortured","torture","terror","terrorist","hijack","hijacking",
        "strangled","strangling","choking","choked","poisoned","poison","acid","shoot","shooter",
        "murdered","executed","execution","sniper","grenade","explosive","massacre","genocide",
        "abuse","abused","domestic","beaten","beating","bruised","wounded","injury","injured",
        "forced","coerced","blackmail","extortion","ransom","carjack","carjacking"],
    1: ["theft","stolen","steal","stole","burglary","broke in","break in","damage","vandalism",
        "narcotics","drugs","drug","suspicious","trespassing","trespass","harassment","harassing",
        "harassed","threatening","fight","fighting","argument","dispute","car","vehicle","motorcycle",
        "missing","fraud","criminal","illegal","prowler","intruder","breaking","bully","bullying",
        "bullied","tease","teasing","teased","cyberbullying","stalking","stalked","stalker",
        "threatening","intimidate","intimidation","menacing","threatening","confrontation",
        "shoplifting","pickpocket","scam","swindle","forge","forgery","counterfeit","impersonate",
        "drunk","intoxicated","alcohol","gambling","prostitution","soliciting","looting","robbery",
        "robbing","stealing","cheating","cheated","assault","groping","inappropriate","touching",
        "peeping","voyeur","trespasser","vandal","graffiti","spray paint","window smash",
        "phone stolen","wallet stolen","bag snatched","snatching","chain snatching"],
    0: ["noise","loud","music","disturbance","parking","minor","littering","loitering",
        "jaywalking","complaint","nuisance","stray","dog","children","playing","construction",
        "horn","idle","idling","drilling","neighbor","smoke","smell","speeding","traffic",
        "argument","dispute","quarrel","shouting","yelling","screaming","barking","honking",
        "blocked","obstruction","litter","garbage","trash","mess","dirty","unhygienic"],
}

def predict_severity(audio_class: str, text_description: str) -> dict:
    audio_sev  = AUDIO_SEVERITY.get(audio_class, 1)
    audio_conf = 0.91
    text_lower = text_description.lower().strip()
    text_sev, text_conf, matched = 0, 0.62, []

    for sev in [2, 1, 0]:
        hits = [kw for kw in KEYWORD_SEVERITY[sev] if kw in text_lower]
        if hits:
            matched.extend(hits)
            if sev > text_sev:
                text_sev  = sev
                text_conf = min(0.72 + 0.04 * len(hits), 0.97)

    if not text_lower:
        text_sev, text_conf = audio_sev, 0.60

    final_sev  = max(audio_sev, text_sev)
    final_conf = round((audio_conf * 0.45 + text_conf * 0.55), 2)

    # Word-level attention scores — only highlight crime-relevant words
    word_scores = []
    if text_lower:
        for word in text_lower.split():
            score = 0.0   # default = not a crime word → won't be highlighted
            for sev in [2, 1, 0]:
                if word in KEYWORD_SEVERITY[sev]:
                    score = 0.5 + sev * 0.2  # High=0.9, Medium=0.7, Low=0.5
                    break
            word_scores.append({"word": word, "score": round(score, 2)})

    return {
        "audio_severity":   audio_sev,
        "audio_conf":       audio_conf,
        "text_severity":    text_sev,
        "text_conf":        round(text_conf, 2),
        "final_severity":   final_sev,
        "final_conf":       final_conf,
        "matched_keywords": list(dict.fromkeys(matched))[:6],
        "word_scores":      word_scores,
    }


def get_precautionary_measures(severity: int, audio_class: str = "Unknown") -> list:
    base = {
        0: ["Log incident", "Continue monitoring", "Routine patrol check"],
        1: ["Increase surveillance", "Alert security team", "Notify local authorities", "Document evidence"],
        2: ["Immediate response required", "Alert police/emergency services",
            "Evacuate area if safe", "Activate full emergency protocol", "Preserve crime scene"],
    }
    measures = list(base.get(severity, []))
    if audio_class not in ("Unknown", ""):
        measures.append(f"Audio trigger: {audio_class.replace('_',' ').title()}")
    return measures


def incidents_to_df(incidents):
    df = pd.DataFrame(incidents)
    if df.empty: return df
    df["timestamp"]      = pd.to_datetime(df["timestamp"])
    df["severity_label"] = df["severity"].map(SEVERITY_LABELS)
    df["color"]          = df["severity"].map(SEVERITY_COLORS)
    return df


def build_demo_incidents():
    specs = [
        {"audio":"gun_shot","loc":0,"mins_ago":15},{"audio":"siren","loc":1,"mins_ago":30},
        {"audio":"drilling","loc":7,"mins_ago":45},{"audio":"engine_idling","loc":4,"mins_ago":60},
        {"audio":"gun_shot","loc":11,"mins_ago":120},{"audio":"siren","loc":8,"mins_ago":180},
        {"audio":"car_horn","loc":2,"mins_ago":75},{"audio":"dog_bark","loc":5,"mins_ago":90},
        {"audio":"jackhammer","loc":6,"mins_ago":105},{"audio":"car_horn","loc":9,"mins_ago":150},
        {"audio":"dog_bark","loc":12,"mins_ago":200},{"audio":"jackhammer","loc":3,"mins_ago":240},
        {"audio":"car_horn","loc":14,"mins_ago":300},{"audio":"children_playing","loc":2,"mins_ago":20},
        {"audio":"street_music","loc":5,"mins_ago":50},{"audio":"air_conditioner","loc":9,"mins_ago":80},
        {"audio":"children_playing","loc":13,"mins_ago":130},{"audio":"street_music","loc":10,"mins_ago":160},
        {"audio":"air_conditioner","loc":6,"mins_ago":220},{"audio":"children_playing","loc":3,"mins_ago":270},
    ]
    incidents = []
    for i, s in enumerate(specs):
        audio = s["audio"]; severity = AUDIO_SEVERITY[audio]
        loc = VIZAG_LOCATIONS[s["loc"]]
        text_cat = TEXT_SEVERITY[severity][i % len(TEXT_SEVERITY[severity])]
        conf = round(0.78 + 0.04 * (i % 6), 2)
        incidents.append({
            "id": f"INC-{i+1:03d}", "location": loc["name"],
            "lat": loc["lat"], "lon": loc["lon"],
            "timestamp": (datetime.now() - pd.Timedelta(minutes=s["mins_ago"])).strftime("%Y-%m-%d %H:%M:%S"),
            "audio_class": audio, "text_category": text_cat,
            "text_content": INCIDENT_DESCRIPTIONS[audio],
            "severity": severity, "confidence": conf,
            "measures": get_precautionary_measures(severity, audio),
        })
    return incidents


DEMO_INCIDENTS = build_demo_incidents()

DARK_AXIS = dict(
    showgrid=True, gridcolor="#2d2d4e", gridwidth=1,
    zeroline=True, zerolinecolor="#3d3d5e",
    linecolor="#444466", linewidth=1,
    tickfont=dict(color="#a0a0c0"),
    title_font=dict(color="#c0c0e0"),
)


def create_severity_chart(incidents):
    df = incidents_to_df(incidents)
    if df.empty: return px.bar(title="No data")
    counts = df["severity_label"].value_counts().reindex(["Low","Medium","High"],fill_value=0).reset_index()
    counts.columns = ["Severity","Count"]
    fig = px.bar(counts, x="Severity", y="Count", color="Severity",
        color_discrete_map={"Low":"#4CAF50","Medium":"#FF9800","High":"#F44336"}, text="Count")
    fig.update_layout(title=dict(text="Severity Distribution",font=dict(color="#c0c0e0",size=14)),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#a0a0c0"}, xaxis=dict(**DARK_AXIS), yaxis=dict(**DARK_AXIS),
        margin=dict(t=40,b=20))
    fig.update_traces(textfont_color="#fff", textposition="outside",
                      marker_line_color="rgba(255,255,255,0.1)", marker_line_width=1)
    return fig


def create_map(incidents):
    df = incidents_to_df(incidents)
    if df.empty:
        return px.scatter_mapbox(zoom=10,center={"lat":17.6868,"lon":83.2185},mapbox_style="open-street-map")
    fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="severity_label",
        color_discrete_map={"Low":"#4CAF50","Medium":"#FF9800","High":"#F44336"},
        size=[16]*len(df), hover_name="id",
        hover_data=["location","timestamp","audio_class","confidence"],
        zoom=10, center={"lat":17.6868,"lon":83.2185}, mapbox_style="open-street-map")
    fig.update_layout(title=dict(text="Geographic Distribution",font=dict(color="#c0c0e0",size=14)),
        margin={"r":0,"t":30,"l":0,"b":0}, legend_title="Severity", height=450,
        paper_bgcolor="rgba(0,0,0,0)")
    fig.update_traces(marker=dict(opacity=0.9))
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
        font={"color":"#a0a0c0"}, xaxis=dict(**DARK_AXIS), yaxis=dict(**DARK_AXIS),
        legend_title="Severity", title_font=dict(color="#c0c0e0",size=14),
        margin=dict(t=40,b=20))
    fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
    return fig


def create_audio_chart(incidents):
    df = incidents_to_df(incidents)
    if df.empty: return px.bar(title="No data")
    counts = df["audio_class"].value_counts().reset_index()
    counts.columns = ["Audio Class","Count"]
    counts["Audio Class"] = counts["Audio Class"].str.replace("_"," ").str.title()
    fig = px.bar(counts, x="Count", y="Audio Class", orientation="h", color="Audio Class",
        color_discrete_map={
            "Gun Shot":"#F44336","Siren":"#F44336","Drilling":"#F44336","Engine Idling":"#F44336",
            "Car Horn":"#FF9800","Dog Bark":"#FF9800","Jackhammer":"#FF9800",
            "Children Playing":"#4CAF50","Street Music":"#4CAF50","Air Conditioner":"#4CAF50",
        },
        title="Audio Class Distribution")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#a0a0c0"}, xaxis=dict(**DARK_AXIS), yaxis=dict(**DARK_AXIS),
        showlegend=False, title_font=dict(color="#c0c0e0",size=14), margin=dict(t=40,b=20))
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], title="Crime Severity Monitor")
server = app.server
SEV_BADGE_COLOR = {0:"success", 1:"warning", 2:"danger"}
SEV_BG = {0:"#0d1f0d", 1:"#1f1800", 2:"#1f0808"}

app.layout = dbc.Container([

    # Header
    dbc.Row(dbc.Col(html.Div([
        html.H1("Multimodal Crime Severity Dashboard", className="text-center my-4",
                style={"color":ACCENT2,"fontWeight":"800","letterSpacing":"0.02em",
                       "textShadow":f"0 0 30px {ACCENT2}55"}),
        html.P("Real-time detection using audio & text modalities | Locations simulated for demo purposes",
               className="text-center text-muted mb-1"),
        html.P("CNN-BiLSTM (Audio) + DistilBERT (Text) | Accuracy: 88.29% | F1: 0.86+ | Deployed on HuggingFace Spaces",
               className="text-center mb-4",
               style={"color":"#a78bfa","fontSize":"0.88em","fontStyle":"italic"}),
    ], style={"background":HEADER_GRAD,"borderRadius":"12px","padding":"8px",
              "marginBottom":"8px","borderBottom":f"1px solid {BORDER_COL}"}))),

    # Stats cards
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H2(str(len(DEMO_INCIDENTS)), className="text-center mb-1",
                    style={"color":ACCENT2,"fontWeight":"800","fontSize":"2.2rem"}),
            html.P("Total Incidents", className="text-center mb-0",
                   style={"color":TEXT_MUTED,"fontSize":"0.8em","letterSpacing":"0.1em","textTransform":"uppercase"}),
        ]), style={"background":CARD_BG,"border":f"1px solid {BORDER_COL}",
                   "borderTop":f"3px solid {ACCENT2}","borderRadius":"10px"}), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H2(str(sum(1 for i in DEMO_INCIDENTS if i["severity"]==2)),
                    className="text-center mb-1",
                    style={"color":"#F44336","fontWeight":"800","fontSize":"2.2rem"}),
            html.P("High Severity", className="text-center mb-0",
                   style={"color":TEXT_MUTED,"fontSize":"0.8em","letterSpacing":"0.1em","textTransform":"uppercase"}),
        ]), style={"background":CARD_BG,"border":f"1px solid {BORDER_COL}",
                   "borderTop":"3px solid #F44336","borderRadius":"10px"}), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H2(str(sum(1 for i in DEMO_INCIDENTS if i["severity"]==1)),
                    className="text-center mb-1",
                    style={"color":"#FF9800","fontWeight":"800","fontSize":"2.2rem"}),
            html.P("Medium Severity", className="text-center mb-0",
                   style={"color":TEXT_MUTED,"fontSize":"0.8em","letterSpacing":"0.1em","textTransform":"uppercase"}),
        ]), style={"background":CARD_BG,"border":f"1px solid {BORDER_COL}",
                   "borderTop":"3px solid #FF9800","borderRadius":"10px"}), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H2(str(sum(1 for i in DEMO_INCIDENTS if i["severity"]==0)),
                    className="text-center mb-1",
                    style={"color":"#4CAF50","fontWeight":"800","fontSize":"2.2rem"}),
            html.P("Low Severity", className="text-center mb-0",
                   style={"color":TEXT_MUTED,"fontSize":"0.8em","letterSpacing":"0.1em","textTransform":"uppercase"}),
        ]), style={"background":CARD_BG,"border":f"1px solid {BORDER_COL}",
                   "borderTop":"3px solid #4CAF50","borderRadius":"10px"}), width=3),
    ], className="mb-4"),

    # Live Inference Panel
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader([
            html.Span("🤖 ", style={"fontSize":"1.2em"}),
            html.Strong("Live Multimodal Inference", style={"color":"#ffffff","fontSize":"1.1em"}),
            html.Span(" — Select audio class + describe the incident to get a real-time severity prediction",
                      style={"color":"#adb5bd","fontSize":"0.85em","marginLeft":"8px"}),
        ], style={"background":ACCENT_GRAD,"borderBottom":f"1px solid {ACCENT}"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("🔊 Audio Class Detected",
                               style={"color":TEXT_PRIMARY,"fontWeight":"bold","marginBottom":"8px"}),
                    dcc.Dropdown(id="inference-audio",
                        options=[{"label":k.replace("_"," ").title(),"value":k} for k in AUDIO_SEVERITY.keys()],
                        value="gun_shot", clearable=False, style={"color":"#000","borderRadius":"8px"}),
                    html.Small("Select the audio class detected at the scene",
                               style={"color":TEXT_MUTED,"marginTop":"4px","display":"block"}),
                ], width=4),
                dbc.Col([
                    html.Label("📝 Crime Description",
                               style={"color":TEXT_PRIMARY,"fontWeight":"bold","marginBottom":"8px"}),
                    dbc.Textarea(id="inference-text",
                        placeholder="Describe the incident... e.g. 'Gunshot heard near ATM, suspect armed with weapon'",
                        value="",
                        style={"background":SURFACE,"color":TEXT_PRIMARY,
                               "border":f"1px solid {BORDER_COL}","borderRadius":"8px",
                               "height":"80px","resize":"none"}),
                    html.Small("Type keywords like: gunshot, robbery, assault, theft, drugs, noise...",
                               style={"color":TEXT_MUTED,"marginTop":"4px","display":"block"}),
                ], width=5),
                dbc.Col([
                    html.Label("⚡ Predict",
                               style={"color":TEXT_PRIMARY,"fontWeight":"bold","marginBottom":"8px"}),
                    dbc.Button("🔍 Classify Severity", id="inference-btn", color="primary",
                        className="w-100 mb-2",
                        style={"background":f"linear-gradient(135deg,{ACCENT},{ACCENT2})",
                               "border":"none","fontWeight":"bold","height":"40px",
                               "borderRadius":"8px","boxShadow":f"0 4px 15px {ACCENT}44"}),
                    dbc.Button("✕ Clear", id="inference-clear", color="secondary", outline=True,
                        className="w-100", size="sm",
                        style={"borderRadius":"8px","borderColor":BORDER_COL}),
                ], width=3),
            ], className="mb-3"),
            html.Div(id="inference-result"),
        ], style={"background":CARD_BG}),
    ], className="shadow mb-4", style={"border":f"1px solid {ACCENT}","borderRadius":"12px"}))),

    # Confidence slider
    dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
        html.Label("🎚️ Confidence Threshold Filter",
                   style={"color":TEXT_PRIMARY,"fontWeight":"bold","fontSize":"1.1em"}),
        dcc.Slider(id="confidence-slider", min=0.0, max=1.0, step=0.05, value=0.5,
            marks={0:"0%",0.25:"25%",0.5:"50%",0.75:"75%",1.0:"100%"},
            tooltip={"placement":"bottom","always_visible":True}),
        html.P("Only show incidents at or above this confidence level",
               style={"color":TEXT_MUTED,"fontSize":"0.85em","marginTop":"8px"}),
    ]), style={"background":CARD_BG,"border":f"1px solid {BORDER_COL}","borderRadius":"10px"},
       className="shadow mb-4"))),

    # Charts
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="severity-chart")),
            style={"background":CARD_BG,"border":f"1px solid {BORDER_COL}","borderRadius":"10px"},
            className="shadow"), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="incident-map")),
            style={"background":CARD_BG,"border":f"1px solid {BORDER_COL}","borderRadius":"10px"},
            className="shadow"), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="timeseries-chart")),
            style={"background":CARD_BG,"border":f"1px solid {BORDER_COL}","borderRadius":"10px"},
            className="shadow"), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="audio-chart")),
            style={"background":CARD_BG,"border":f"1px solid {BORDER_COL}","borderRadius":"10px"},
            className="shadow"), width=6),
    ], className="mb-4"),

    # Alerts
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("⚠ High Severity Alerts",
            style={"background":"linear-gradient(90deg,#7f1d1d,#991b1b)",
                   "color":"#fff","fontWeight":"bold","borderRadius":"10px 10px 0 0"}),
        dbc.CardBody(id="high-severity-list"),
    ], style={"background":CARD_BG,"border":"1px solid #F4433655","borderRadius":"10px"},
       className="shadow mb-4"))),

    # Table
    dbc.Row(dbc.Col(dbc.Card([
        dbc.CardHeader("📋 All Incidents",
            style={"background":f"linear-gradient(90deg,#1e1b4b,{ACCENT}44)",
                   "color":"#fff","fontWeight":"bold","borderRadius":"10px 10px 0 0"}),
        dbc.CardBody(id="incidents-table"),
    ], style={"background":CARD_BG,"border":f"1px solid {BORDER_COL}","borderRadius":"10px"},
       className="shadow"))),

    dcc.Store(id="stored-incidents",   data=DEMO_INCIDENTS),
    dcc.Store(id="filtered-incidents", data=DEMO_INCIDENTS),
    dcc.Interval(id="interval-component", interval=30_000, n_intervals=0),

    dbc.Modal([
        dbc.ModalHeader(html.H4(id="modal-title"),
            style={"background":ACCENT_GRAD,"color":TEXT_PRIMARY,"borderBottom":f"1px solid {ACCENT}"}),
        dbc.ModalBody(id="modal-content", style={"background":"#0d0d1a","color":TEXT_PRIMARY}),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto", color="secondary",
                       style={"borderRadius":"8px"}),
            style={"background":CARD_BG,"borderTop":f"1px solid {BORDER_COL}"}),
    ], id="incident-modal", size="lg"),

], fluid=True, style={"background":"#080810","minHeight":"100vh","padding":"20px","color":TEXT_PRIMARY})


# ── Inference callback ────────────────────────────────────────────────────────
@callback(Output("inference-result","children"),
          Input("inference-btn","n_clicks"),
          State("inference-audio","value"), State("inference-text","value"),
          prevent_initial_call=True)
def run_inference(n_clicks, audio_class, text_description):
    if not audio_class:
        return html.P("Please select an audio class.", style={"color":TEXT_MUTED})
    text   = text_description.strip() if text_description else ""
    result = predict_severity(audio_class, text)
    sev       = result["final_severity"]
    sev_label = SEVERITY_LABELS[sev]
    sev_color = SEVERITY_COLORS[sev]
    sev_badge = SEV_BADGE_COLOR[sev]
    measures  = get_precautionary_measures(sev, audio_class)

    # Word attention visualization — only show when text found matching keywords
    word_scores  = result.get("word_scores", [])
    text_sev     = result.get("text_severity", 0)
    has_keywords = len(result.get("matched_keywords", [])) > 0
    attention_section = []
    if word_scores and (text_sev > 0 or has_keywords):
        word_spans = []
        for ws in word_scores:
            score = ws["score"]
            # score=0 means not a crime word → show as plain grey text
            if score == 0:
                bg = "transparent"; col = "#606080"
                bold = False
            elif text_sev == 2:
                if score >= 0.9:
                    bg = "rgba(244,67,54,0.85)"; col = "#fff"; bold = True
                elif score >= 0.7:
                    bg = "rgba(255,152,0,0.75)"; col = "#fff"; bold = True
                else:
                    bg = "rgba(100,100,160,0.3)"; col = "#c0c0d0"; bold = False
            elif text_sev == 1:
                if score >= 0.7:
                    bg = "rgba(255,152,0,0.85)"; col = "#fff"; bold = True
                elif score >= 0.5:
                    bg = "rgba(255,193,7,0.65)"; col = "#333"; bold = True
                else:
                    bg = "rgba(100,100,160,0.2)"; col = "#c0c0d0"; bold = False
            else:
                if score >= 0.5:
                    bg = "rgba(76,175,80,0.65)"; col = "#fff"; bold = True
                else:
                    bg = "transparent"; col = "#606080"; bold = False

            # Show word + percentage only for crime keywords (score > 0)
            display_text = f"{ws['word']} {round(score*100)}%" if score > 0 else ws['word']

            word_spans.append(html.Span(display_text+" ",
                style={"background":bg,"color":col,"padding":"2px 7px",
                       "borderRadius":"4px","margin":"2px","display":"inline-block",
                       "fontSize":"0.88em","fontWeight":"600" if bold else "normal"}))

        # Legend changes based on text severity
        if text_sev == 2:
            legend = "🔴 High influence  🟠 Medium  ⬜ Low  |  Colors reflect text severity prediction"
        elif text_sev == 1:
            legend = "🟠 High influence  🟡 Medium  ⬜ Low  |  Colors reflect text severity prediction"
        else:
            legend = "🟢 Words shown — text predicts Low severity  |  Audio drove the final prediction"

        attention_section = [
            html.Hr(style={"borderColor":BORDER_COL,"margin":"12px 0"}),
            html.P("🔍 Word Attention — Text Modality",
                   style={"color":TEXT_MUTED,"fontSize":"0.7em","letterSpacing":"0.15em",
                          "marginBottom":"8px","fontWeight":"bold","textTransform":"uppercase"}),
            html.Div(word_spans,
                     style={"lineHeight":"2.4","padding":"10px","background":SURFACE,
                            "borderRadius":"8px","border":f"1px solid {BORDER_COL}"}),
            html.P(legend,
                   style={"color":TEXT_MUTED,"fontSize":"0.72em","marginTop":"6px","fontStyle":"italic"}),
        ]

    keyword_badges = [
        html.Span(kw, className="badge me-1", style={"background":ACCENT,"fontSize":"0.75em"})
        for kw in result["matched_keywords"]
    ] or [html.Span("none", style={"color":TEXT_MUTED,"fontSize":"0.78em"})]

    return dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.P("FINAL PREDICTION",
                       style={"color":TEXT_MUTED,"fontSize":"0.7em","letterSpacing":"0.15em",
                              "marginBottom":"4px","fontWeight":"bold"}),
                html.H2(sev_label,
                        style={"color":sev_color,"fontWeight":"800","fontSize":"3rem",
                               "marginBottom":"4px","textShadow":f"0 0 20px {sev_color}88"}),
                html.P(f"Confidence: {result['final_conf']:.0%}",
                       style={"color":TEXT_PRIMARY,"fontSize":"1em"}),
                dbc.Progress(value=result["final_conf"]*100, color=sev_badge,
                             striped=True, animated=True,
                             style={"height":"10px","marginTop":"8px","borderRadius":"5px"}),
            ], width=4, style={"borderRight":f"1px solid {BORDER_COL}","paddingRight":"20px"}),

            dbc.Col([
                html.P("MODALITY BREAKDOWN",
                       style={"color":TEXT_MUTED,"fontSize":"0.7em","letterSpacing":"0.15em",
                              "marginBottom":"12px","fontWeight":"bold"}),
                dbc.Row([
                    dbc.Col([
                        html.Div([html.Span("🔊 Audio: ",style={"color":TEXT_MUTED,"fontSize":"0.85em"}),
                                  html.Span(audio_class.replace("_"," ").title(),
                                            style={"color":TEXT_PRIMARY,"fontWeight":"bold","fontSize":"0.85em"})]),
                        html.Div([html.Span("Severity: ",style={"color":TEXT_MUTED,"fontSize":"0.82em"}),
                                  html.Span(SEVERITY_LABELS[result["audio_severity"]],
                                            className=f"badge bg-{SEV_BADGE_COLOR[result['audio_severity']]}",
                                            style={"fontSize":"0.75em"}),
                                  html.Span(f"  {result['audio_conf']:.0%}",
                                            style={"color":TEXT_MUTED,"fontSize":"0.78em"})],
                                 style={"marginTop":"4px"}),
                    ], width=6),
                    dbc.Col([
                        html.Div([html.Span("📝 Text: ",style={"color":TEXT_MUTED,"fontSize":"0.85em"}),
                                  html.Span(f"{len(text.split())} words" if text else "No input",
                                            style={"color":TEXT_PRIMARY,"fontWeight":"bold","fontSize":"0.85em"})]),
                        html.Div([html.Span("Severity: ",style={"color":TEXT_MUTED,"fontSize":"0.82em"}),
                                  html.Span(SEVERITY_LABELS[result["text_severity"]],
                                            className=f"badge bg-{SEV_BADGE_COLOR[result['text_severity']]}",
                                            style={"fontSize":"0.75em"}),
                                  html.Span(f"  {result['text_conf']:.0%}",
                                            style={"color":TEXT_MUTED,"fontSize":"0.78em"})],
                                 style={"marginTop":"4px"}),
                    ], width=6),
                ]),
                html.Hr(style={"borderColor":BORDER_COL,"margin":"10px 0"}),
                html.Div([html.Span("Fusion: ",style={"color":TEXT_MUTED,"fontSize":"0.8em"}),
                          html.Span("max(audio, text) — Conservative",
                                    style={"color":"#a78bfa","fontSize":"0.8em","fontStyle":"italic"})]),
                html.Div([html.Span("Keywords: ",style={"color":TEXT_MUTED,"fontSize":"0.78em"}),
                          *keyword_badges], style={"marginTop":"6px"}),
                *attention_section,
            ], width=5),

            dbc.Col([
                html.P("RECOMMENDED ACTIONS",
                       style={"color":TEXT_MUTED,"fontSize":"0.7em","letterSpacing":"0.15em",
                              "marginBottom":"8px","fontWeight":"bold"}),
                html.Ul([html.Li(m, style={"color":TEXT_PRIMARY,"fontSize":"0.82em","marginBottom":"4px"})
                         for m in measures[:4]], style={"paddingLeft":"16px"}),
            ], width=3, style={"borderLeft":f"1px solid {BORDER_COL}","paddingLeft":"20px"}),
        ]),
    ]), style={"background":SURFACE,"border":f"1px solid {sev_color}55",
               "borderLeft":f"4px solid {sev_color}","borderRadius":"10px",
               "boxShadow":f"0 4px 20px {sev_color}22"})


@callback(Output("inference-text","value"), Output("inference-audio","value"),
          Output("inference-result","children",allow_duplicate=True),
          Input("inference-clear","n_clicks"), prevent_initial_call=True)
def clear_inference(n): return "", "gun_shot", ""


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
    high = sorted([i for i in incidents if i["severity"]==2],key=lambda x:x["confidence"],reverse=True)
    if not high:
        return html.P("No high severity incidents above threshold.",
                      className="text-center p-3", style={"color":TEXT_MUTED})
    return [dbc.Card(dbc.CardBody([
        dbc.Row([
            # Left — incident info
            dbc.Col([
                html.Div([
                    html.Span(i["id"],
                              style={"background":"#F44336","color":"#fff","padding":"2px 10px",
                                     "borderRadius":"4px","fontSize":"0.78em","fontWeight":"bold",
                                     "marginRight":"8px"}),
                    html.Span(i["location"],
                              style={"color":TEXT_PRIMARY,"fontWeight":"bold","fontSize":"0.95em"}),
                ], style={"marginBottom":"6px"}),
                html.Div([
                    html.Span("🕐 ", style={"color":TEXT_MUTED}),
                    html.Span(i["timestamp"], style={"color":TEXT_MUTED,"fontSize":"0.8em","marginRight":"12px"}),
                    html.Span("🔊 ", style={"color":TEXT_MUTED}),
                    html.Span(i["audio_class"].replace("_"," ").title(),
                              style={"color":"#F44336","fontSize":"0.8em","fontWeight":"bold"}),
                ], style={"marginBottom":"8px"}),
                html.P(i["text_content"],
                       style={"color":"#c0c0d0","fontSize":"0.85em","marginBottom":"8px",
                              "borderLeft":"2px solid #F44336","paddingLeft":"8px"}),
                html.Div([
                    html.Span(f"Confidence: {i['confidence']:.0%}",
                              style={"color":"#F44336","fontWeight":"bold","fontSize":"0.82em","marginRight":"12px"}),
                    html.Span(i["text_category"],
                              style={"background":"#2d1515","color":"#ff8080","padding":"2px 8px",
                                     "borderRadius":"4px","fontSize":"0.78em"}),
                ]),
            ], width=8),
            # Right — actions
            dbc.Col([
                html.P("⚡ Recommended Actions",
                       style={"color":"#ffc107","fontWeight":"bold","fontSize":"0.8em",
                              "marginBottom":"6px","textTransform":"uppercase","letterSpacing":"0.05em"}),
                html.Ul([
                    html.Li(m, style={"color":TEXT_PRIMARY,"fontSize":"0.78em","marginBottom":"3px"})
                    for m in i["measures"][:3]
                ], style={"paddingLeft":"14px","marginBottom":"10px"}),
                dbc.Button([
                    html.Span("View Details"),
                ], id={"type":"high-button","index":i["id"]},
                   color="danger", size="sm",
                   style={"borderRadius":"6px","fontWeight":"bold","width":"100%",
                          "background":"linear-gradient(135deg,#c62828,#F44336)",
                          "border":"none"}),
            ], width=4, style={"borderLeft":"1px solid #F4433333","paddingLeft":"16px"}),
        ]),
    ], style={"background":"linear-gradient(135deg,#1a0808,#200d0d)","borderRadius":"8px","padding":"16px"}),
    style={"border":"1px solid #F4433644","borderLeft":"4px solid #F44336",
           "borderRadius":"10px","marginBottom":"14px",
           "boxShadow":"0 2px 12px rgba(244,67,54,0.15)"})
    for i in high]


@callback(Output("incidents-table","children"), Input("filtered-incidents","data"))
def upd_table(incidents):
    sinc = sorted(incidents, key=lambda x: x["timestamp"], reverse=True)
    header = html.Thead(html.Tr(
        [html.Th(c, style={"color":"#c0c0e0","fontWeight":"bold","fontSize":"0.82em",
                            "textTransform":"uppercase","letterSpacing":"0.08em"})
         for c in ["ID","Location","Time","Audio","Category","Severity","Confidence","Action"]],
        style={"background":f"linear-gradient(90deg,#1e1b4b,{ACCENT}33)"}))
    rows = [html.Tr([
        html.Td(i["id"], style={"color":ACCENT2,"fontWeight":"600","fontSize":"0.88em"}),
        html.Td(i["location"], style={"color":TEXT_PRIMARY,"fontSize":"0.88em"}),
        html.Td(i["timestamp"], style={"color":TEXT_MUTED,"fontSize":"0.8em"}),
        html.Td(i["audio_class"].replace("_"," ").title(), style={"color":"#c0c0d0","fontSize":"0.85em"}),
        html.Td(i["text_category"], style={"color":"#c0c0d0","fontSize":"0.85em"}),
        html.Td(html.Span(SEVERITY_LABELS[i["severity"]],
                          className=f"badge bg-{SEV_BADGE_COLOR[i['severity']]}",
                          style={"fontSize":"0.85em","fontWeight":"bold"})),
        html.Td(f"{i['confidence']:.0%}", style={"color":ACCENT2,"fontWeight":"600","fontSize":"0.88em"}),
        html.Td(dbc.Button("Details", id={"type":"table-button","index":i["id"]},
                           color="info", size="sm", outline=True,
                           style={"borderRadius":"6px","fontSize":"0.78em"})),
    ], style={"background":SEV_BG[i["severity"]],"borderBottom":f"1px solid {BORDER_COL}"})
    for i in sinc]
    return dbc.Table([header, html.Tbody(rows)],
                     bordered=False, hover=True, responsive=True,
                     style={"borderColor":BORDER_COL})


def modal_body(inc):
    sev_color = SEVERITY_COLORS[inc["severity"]]
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.P([html.Strong("📍 Location: "), inc["location"]], style={"color":TEXT_PRIMARY}),
                html.P([html.Strong("🕐 Time: "), inc["timestamp"]], style={"color":TEXT_PRIMARY}),
                html.P([html.Strong("🔊 Audio: "), inc["audio_class"].replace("_"," ").title()], style={"color":TEXT_PRIMARY}),
                html.P([html.Strong("📋 Category: "), inc["text_category"]], style={"color":TEXT_PRIMARY}),
            ], width=6),
            dbc.Col([
                html.P("Severity Assessment", style={"color":ACCENT2,"fontWeight":"bold"}),
                dbc.Progress(value=inc["confidence"]*100,
                             color=["success","warning","danger"][inc["severity"]],
                             striped=True, animated=True, style={"height":"20px","borderRadius":"10px"}),
                html.P(f"{SEVERITY_LABELS[inc['severity']]} — {inc['confidence']:.1%} confidence",
                       style={"color":sev_color,"fontWeight":"bold","marginTop":"6px"}),
            ], width=6),
        ]),
        html.Hr(style={"borderColor":BORDER_COL}),
        html.P(html.Strong("Description"), style={"color":ACCENT2}),
        html.P(inc["text_content"], style={"color":"#c0c0d0"}),
        html.P(html.Strong("Recommended Measures"), style={"color":ACCENT2}),
        html.Ul([html.Li(m, style={"color":TEXT_PRIMARY,"marginBottom":"4px"}) for m in inc["measures"]]),
        html.P(html.Strong("Location Map"), style={"color":ACCENT2}),
        dcc.Graph(figure=px.scatter_mapbox(pd.DataFrame([inc]),lat="lat",lon="lon",
                                           zoom=14,mapbox_style="open-street-map").update_layout(
                                           paper_bgcolor="#0d0d1a",
                                           margin={"r":0,"t":0,"l":0,"b":0},
                                           height=280)),
    ], style={"color":TEXT_PRIMARY})


@callback(
    [Output("incident-modal","is_open",allow_duplicate=True),
     Output("modal-title","children",allow_duplicate=True),
     Output("modal-content","children",allow_duplicate=True)],
    Input({"type":"high-button","index":dash.ALL},"n_clicks"),
    State("stored-incidents","data"), State("incident-modal","is_open"),
    prevent_initial_call=True)
def modal_high(n_clicks, incidents, is_open):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks): return dash.no_update,dash.no_update,dash.no_update
    inc_id = eval(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    inc = next((i for i in incidents if i["id"]==inc_id), None)
    return (True, f"🔍 {inc['id']}", modal_body(inc)) if inc else (is_open,dash.no_update,dash.no_update)


@callback(
    [Output("incident-modal","is_open"),
     Output("modal-title","children"),
     Output("modal-content","children")],
    Input({"type":"table-button","index":dash.ALL},"n_clicks"),
    State("stored-incidents","data"), State("incident-modal","is_open"),
    prevent_initial_call=True)
def modal_table(n_clicks, incidents, is_open):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks): return dash.no_update,dash.no_update,dash.no_update
    inc_id = eval(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    inc = next((i for i in incidents if i["id"]==inc_id), None)
    return (True, f"🔍 {inc['id']}", modal_body(inc)) if inc else (is_open,dash.no_update,dash.no_update)


@callback(Output("incident-modal","is_open",allow_duplicate=True),
          Input("close-modal","n_clicks"), State("incident-modal","is_open"),
          prevent_initial_call=True)
def close_modal(n, is_open): return not is_open if n else is_open


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))