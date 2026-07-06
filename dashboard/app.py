import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# On Streamlit Cloud, secrets are set in the app's Settings > Secrets, not as
# real environment variables - so pull GROQ_API_KEY from st.secrets into the
# environment if it's there, and llm_report.py's generate_report(backend="auto")
# will pick Groq automatically. Locally (no secrets.toml), this is a no-op and
# it falls back to Ollama.
import os
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    pass

from llm_report import generate_report, CLASS_RELIABILITY  # noqa: E402

st.set_page_config(page_title="TrustSOC AI", layout="wide", page_icon="\U0001F6E1\uFE0F")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG = "#0A0E12"
SURFACE = "#131920"
BORDER = "#232B35"
TEXT = "#E6EDF3"
MUTED = "#8B98A5"
TEAL = "#2DD4BF"      # primary accent: verification / trust
DANGER = "#F87171"    # high-severity attack classes
WARNING = "#FBBF24"   # low confidence / caution
SUCCESS = "#34D399"   # benign / high reliability

SEVERITY = {
    "Benign": SUCCESS,
    "PortScan": WARNING,
    "Bot": DANGER,
    "DDoS": DANGER,
    "DoS Hulk": DANGER,
    "DoS GoldenEye": DANGER,
    "DoS slowloris": DANGER,
    "DoS Slowhttptest": DANGER,
    "FTP-Patator": WARNING,
    "SSH-Patator": WARNING,
    "Web Attack": DANGER,
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
.stApp {{
    background-color: {BG};
}}
[data-testid="stSidebar"] {{
    background-color: {SURFACE};
    border-right: 1px solid {BORDER};
}}
.mono {{
    font-family: 'JetBrains Mono', monospace;
}}
.console-header {{
    font-family: 'JetBrains Mono', monospace;
    color: {TEAL};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.75rem;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 6px;
    margin-bottom: 12px;
}}
.card {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 18px 20px;
    margin-bottom: 16px;
}}
.badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 999px;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
PRED_PATH = ROOT / "schema" / "sample_predictions.json"
with open(PRED_PATH) as f:
    records = json.load(f)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="console-header">TrustSOC AI // Console</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="mono" style="color:{MUTED}">'
        f'{len(records)} flows loaded from local pipeline'
        f'</span>', unsafe_allow_html=True
    )
    st.write("")

    class_counts = pd.Series([r["predicted_class"] for r in records]).value_counts()
    for cls, count in class_counts.items():
        color = SEVERITY.get(cls, MUTED)
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
            f'<span class="mono" style="color:{TEXT};font-size:0.82rem;">'
            f'<span style="color:{color};">\u25CF</span> {cls}</span>'
            f'<span class="mono" style="color:{MUTED};font-size:0.82rem;">{count}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.write("---")
    flow_options = {
        f"#{r['flow_id']}  \u2022  {r['predicted_class']}  \u2022  {r['confidence']:.0%}": r
        for r in records
    }
    selected_label = st.selectbox("Select flow", list(flow_options.keys()))
    record = flow_options[selected_label]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
severity_color = SEVERITY.get(record["predicted_class"], MUTED)
st.markdown(f"""
<div style="display:flex;align-items:baseline;gap:14px;margin-bottom:4px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:700;color:{TEXT};">
        {record['predicted_class']}
    </span>
    <span class="badge" style="background-color:{severity_color}22;color:{severity_color};border:1px solid {severity_color}55;">
        FLOW #{record['flow_id']}
    </span>
</div>
<div class="mono" style="color:{MUTED};font-size:0.85rem;margin-bottom:22px;">
    single-flow inspection &middot; SHAP-grounded &middot; local LLM report
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Trust Score gauge (signature element) + SHAP chart
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<div class="console-header">Trust Score</div>', unsafe_allow_html=True)

    flow_conf = record["confidence"]
    reliability = CLASS_RELIABILITY.get(record["predicted_class"], {})
    precision = reliability.get("precision", None)

    # Trust score fuses THIS flow's confidence with the class's historical
    # precision -- deliberately not just the confidence alone, since that's
    # the whole point of separating these two signals.
    trust_score = flow_conf * precision if precision is not None else flow_conf

    def hex_to_rgba(hex_color, alpha=0.2):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=trust_score * 100,
        number={"suffix": "%", "font": {"color": TEXT, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
            "bar": {"color": TEAL},
            "bgcolor": SURFACE,
            "borderwidth": 1,
            "bordercolor": BORDER,
            "steps": [
                {"range": [0, 50], "color": hex_to_rgba(DANGER)},
                {"range": [50, 80], "color": hex_to_rgba(WARNING)},
                {"range": [80, 100], "color": hex_to_rgba(SUCCESS)},
            ],
        },
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor=BG,
        font={"color": TEXT},
    )
    st.plotly_chart(fig, use_container_width=True)

    precision_str = f"{precision:.1%}" if precision is not None else "n/a"
    st.markdown(
        f'<div class="mono" style="font-size:0.78rem;color:{MUTED};line-height:1.6;">'
        f'flow confidence &nbsp;<span style="color:{TEXT}">{flow_conf:.1%}</span><br>'
        f'class precision &nbsp;<span style="color:{TEXT}">{precision_str}</span><br>'
        f'<span style="color:{MUTED}">trust score = flow confidence \u00d7 class precision</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown('<div class="console-header">SHAP Feature Attribution</div>', unsafe_allow_html=True)

    shap_df = pd.DataFrame(record["top_shap_features"]).sort_values("shap_value")
    bar_fig = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=TEAL,
        text=shap_df["value"].apply(lambda v: f"val: {v:g}"),
        textposition="outside",
        textfont={"color": MUTED, "family": "JetBrains Mono", "size": 11},
    ))
    bar_fig.update_layout(
        height=260,
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={"color": TEXT, "family": "JetBrains Mono"},
        xaxis={"gridcolor": BORDER, "title": "SHAP value"},
        yaxis={"gridcolor": BORDER},
    )
    st.plotly_chart(bar_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Analyst report
# ---------------------------------------------------------------------------
st.markdown('<div class="console-header">Analyst Report</div>', unsafe_allow_html=True)

if st.button("\u25B6 Generate Report", type="primary"):
    with st.spinner("Querying local model (Ollama / llama3.1:8b)..."):
        try:
            report = generate_report(record)
        except Exception as e:
            report = None
            st.error(f"Report generation failed: {e}")
            st.info("Make sure `ollama serve` is running and `llama3.1:8b` is pulled.")

    if report:
        st.markdown(f"""
        <div class="card">
            <div style="color:{TEAL};font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Summary</div>
            <div style="color:{TEXT};margin-bottom:18px;">{report['summary']}</div>

            <div style="color:{TEAL};font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Evidence</div>
            <div style="color:{TEXT};margin-bottom:18px;">{report['evidence_explanation']}</div>

            <div style="color:{TEAL};font-family:'JetBrains Mono',monospace;font-size:0.78rem;
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Confidence Note</div>
            <div style="color:{MUTED};margin-bottom:18px;">{report['confidence_note']}</div>
        </div>
        """, unsafe_allow_html=True)

        col_m, col_a = st.columns(2)
        with col_m:
            st.markdown('<div class="console-header">MITRE ATT&CK</div>', unsafe_allow_html=True)
            if report["mitre_mapping"]:
                for t in report["mitre_mapping"]:
                    st.markdown(
                        f'<div class="card" style="padding:12px 16px;">'
                        f'<span class="mono" style="color:{TEAL};font-weight:600;">{t["technique_id"]}</span> '
                        f'<span style="color:{TEXT};">{t["technique_name"]}</span><br>'
                        f'<span class="mono" style="color:{MUTED};font-size:0.75rem;">{t["tactic"]}</span>'
                        f'</div>', unsafe_allow_html=True
                    )
            else:
                st.markdown(f'<span style="color:{MUTED};">No technique mapped (benign).</span>', unsafe_allow_html=True)

        with col_a:
            st.markdown('<div class="console-header">Recommended Actions</div>', unsafe_allow_html=True)
            for action in report["recommended_actions"]:
                st.markdown(
                    f'<div style="color:{TEXT};margin-bottom:6px;">'
                    f'<span style="color:{TEAL};">&rarr;</span> {action}</div>',
                    unsafe_allow_html=True
                )
else:
    st.markdown(
        f'<div class="card" style="color:{MUTED};text-align:center;padding:30px;">'
        f'Click <b style="color:{TEAL};">Generate Report</b> to run the local LLM on this flow.'
        f'</div>', unsafe_allow_html=True
    )