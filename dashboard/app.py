import json
import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    pass

from llm_report import generate_report, CLASS_RELIABILITY, MITRE_LOOKUP  # noqa: E402
from predict_pipeline import process_uploaded_csv, ModelNotAvailableError  # noqa: E402

st.set_page_config(page_title="TrustSOC AI", layout="wide", page_icon="\U0001F6E1\uFE0F")

# ---------------------------------------------------------------------------
# Design tokens (light theme)
# ---------------------------------------------------------------------------
BG = "#FFFFFF"
SURFACE = "#FFFFFF"
GRID_LINE = "#EEEEF2"
BORDER = "#E5E7EB"
TEXT = "#0B0B0F"
MUTED = "#6B7280"
ACCENT = "#6D5EF7"     # purple - "explain" / verification accent
DANGER = "#DC2626"
WARNING = "#D97706"
SUCCESS = "#16A34A"

SEVERITY = {
    "Benign": ("BENIGN", SUCCESS),
    "PortScan": ("SUSPICIOUS", WARNING),
    "FTP-Patator": ("SUSPICIOUS", WARNING),
    "SSH-Patator": ("SUSPICIOUS", WARNING),
    "Bot": ("CRITICAL", DANGER),
    "DDoS": ("CRITICAL", DANGER),
    "DoS Hulk": ("CRITICAL", DANGER),
    "DoS GoldenEye": ("CRITICAL", DANGER),
    "DoS slowloris": ("CRITICAL", DANGER),
    "DoS Slowhttptest": ("CRITICAL", DANGER),
    "Web Attack": ("CRITICAL", DANGER),
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {TEXT}; }}
.stApp {{
    background-color: {BG};
    background-image:
        linear-gradient(to right, {GRID_LINE} 1px, transparent 1px),
        linear-gradient(to bottom, {GRID_LINE} 1px, transparent 1px);
    background-size: 40px 40px;
}}
.mono {{ font-family: 'JetBrains Mono', monospace; }}
.eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 0.75rem;
    font-weight: 600;
}}
.headline {{
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin: 8px 0 14px 0;
}}
.card {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 20px 22px;
}}
.stat-card {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 18px;
}}
.stat-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {MUTED};
}}
.stat-value {{ font-size: 1.7rem; font-weight: 700; margin-top: 2px; }}
.stat-sub {{ font-size: 0.75rem; color: {MUTED}; }}
.badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 2px 10px;
    border-radius: 999px;
}}
.section-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 6px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
PRED_PATH = ROOT / "schema" / "sample_predictions.json"
with open(PRED_PATH) as f:
    demo_records = json.load(f)

if "uploaded_records" not in st.session_state:
    st.session_state.uploaded_records = None

records = st.session_state.uploaded_records if st.session_state.uploaded_records is not None else demo_records

if st.session_state.get("upload_msg"):
    st.success(st.session_state.upload_msg)
    del st.session_state.upload_msg

if "scan_run" not in st.session_state:
    st.session_state.scan_run = False

# ---------------------------------------------------------------------------
# Top nav
# ---------------------------------------------------------------------------
nav_l, nav_r = st.columns([3, 2])
with nav_l:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:34px;height:34px;background:{ACCENT};border-radius:8px;
                    display:flex;align-items:center;justify-content:center;font-size:1.1rem;">
            \U0001F6E1\uFE0F
        </div>
        <div>
            <div style="font-weight:800;font-size:1.1rem;line-height:1;">
                TrustSOC <span style="color:{ACCENT};">AI</span>
            </div>
            <div class="mono" style="font-size:0.65rem;color:{MUTED};letter-spacing:0.1em;">
                DETECT &middot; EXPLAIN &middot; REPORT
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with nav_r:
    c1, c2 = st.columns([2, 1])
    with c1:
        backend_label = st.selectbox(
            "LLM backend", ["groq (llama-3.1-8b-instant)", "ollama (llama3.1:8b, local)"],
            label_visibility="collapsed",
        )
    with c2:
        st.markdown(
            f'<div style="text-align:right;padding-top:8px;">'
            f'<span style="color:{SUCCESS};">\u25CF</span> '
            f'<span class="mono" style="font-size:0.75rem;color:{MUTED};">OPERATIONAL</span></div>',
            unsafe_allow_html=True,
        )
backend = "groq" if backend_label.startswith("groq") else "ollama"

st.write("")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
hero_l, hero_r = st.columns([3, 2], gap="large")
with hero_l:
    st.markdown('<div class="eyebrow">\u2726 Explainable Intrusion Detection</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="headline">
        Detect threats.<br>
        <span style="color:{ACCENT};">Explain</span> every prediction.<br>
        Report with evidence.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{MUTED};max-width:520px;margin-bottom:18px;">'
        f'TrustSOC AI classifies network flows into DDoS, PortScan, Bot, Web Attack, '
        f'and more &mdash; then grounds every incident report in the exact SHAP evidence '
        f'used by the model. No invented IPs. No black-box verdicts.</div>',
        unsafe_allow_html=True,
    )
    btn_col, upload_col = st.columns([1, 1.4])
    with btn_col:
        if st.button("\u25B6 Run demo scan", type="primary"):
            st.session_state.uploaded_records = None
            st.session_state.scan_run = True
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload CSV", type=["csv"], label_visibility="collapsed"
        )
        if uploaded_file is not None:
            file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_processed_file") != file_signature:
                with st.spinner("Running model + SHAP on uploaded flows..."):
                    try:
                        st.session_state.uploaded_records = process_uploaded_csv(uploaded_file)
                        st.session_state.scan_run = True
                        st.session_state.last_processed_file = file_signature
                        st.session_state.upload_msg = f"Processed {len(st.session_state.uploaded_records)} flows from upload."
                        st.rerun()
                    except ModelNotAvailableError as e:
                        st.warning(str(e))
                    except Exception as e:
                        st.error(f"Could not process file: {e}")

with hero_r:
    st.markdown(f"""
    <div class="card">
        <div class="section-label">Pipeline</div>
        <div style="display:flex;gap:10px;margin-bottom:14px;">
            <span class="mono" style="color:{MUTED};font-size:0.75rem;">01</span>
            <div><b>Ingest</b><br><span style="color:{MUTED};font-size:0.8rem;">CICIDS2017-schema flows</span></div>
        </div>
        <div style="display:flex;gap:10px;margin-bottom:14px;">
            <span class="mono" style="color:{MUTED};font-size:0.75rem;">02</span>
            <div><b>Classify</b><br><span style="color:{MUTED};font-size:0.8rem;">11-class detector (XGBoost)</span></div>
        </div>
        <div style="display:flex;gap:10px;margin-bottom:14px;">
            <span class="mono" style="color:{MUTED};font-size:0.75rem;">03</span>
            <div><b>Explain</b><br><span style="color:{MUTED};font-size:0.8rem;">Per-flow SHAP contributions</span></div>
        </div>
        <div style="display:flex;gap:10px;">
            <span class="mono" style="color:{MUTED};font-size:0.75rem;">04</span>
            <div><b>Report</b><br><span style="color:{MUTED};font-size:0.8rem;">Grounded LLM incident narrative</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Session stats + table (only after "Run demo scan")
# ---------------------------------------------------------------------------
if not st.session_state.scan_run:
    st.markdown(f"""
    <div class="card" style="text-align:center;padding:50px;color:{MUTED};">
        <div class="mono" style="font-size:0.7rem;letter-spacing:0.1em;margin-bottom:8px;">NO SESSION</div>
        <div style="font-weight:700;color:{TEXT};font-size:1.1rem;margin-bottom:6px;">Run the demo scan to begin</div>
        <div style="font-size:0.85rem;">The dashboard will populate with detection results, SHAP evidence,
        and grounded incident reports.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = pd.DataFrame(records)
malicious = df[df["predicted_class"] != "Benign"]
critical = df[df["predicted_class"].map(lambda c: SEVERITY.get(c, ("", MUTED))[0] == "CRITICAL")]
top_attack = malicious["predicted_class"].mode()[0] if not malicious.empty else "-"
avg_conf = df["confidence"].mean()

st.markdown(
    f'<div class="mono" style="font-size:0.75rem;color:{MUTED};margin-bottom:10px;">'
    f'SESSION &middot; <b style="color:{TEXT};">DEMO-{len(df):03d}</b></div>',
    unsafe_allow_html=True,
)

s1, s2, s3, s4, s5 = st.columns(5)
stats = [
    (s1, "TOTAL FLOWS", str(len(df)), "in current session"),
    (s2, "MALICIOUS", str(len(malicious)), f"{len(malicious)/len(df):.0%} of flows"),
    (s3, "CRITICAL", str(len(critical)), "highest severity"),
    (s4, "TOP ATTACK", top_attack, "most frequent class"),
    (s5, "AVG CONFIDENCE", f"{avg_conf:.1%}", "model certainty"),
]
for col, label, value, sub in stats:
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Detection table
# ---------------------------------------------------------------------------
tab_detect, tab_insights = st.tabs(["Detection", "Model Insights"])

with tab_detect:
    st.markdown('<div class="section-label">Live Detection Results &middot; click a row for SHAP evidence</div>', unsafe_allow_html=True)

    display_df = df[["flow_id", "predicted_class", "confidence"]].copy()
    display_df["confidence_pct"] = display_df["confidence"] * 100
    display_df["severity"] = display_df["predicted_class"].map(lambda c: SEVERITY.get(c, ("UNKNOWN", MUTED))[0])
    display_df = display_df[["flow_id", "predicted_class", "confidence_pct", "severity"]]
    display_df.columns = ["Flow ID", "Predicted Class", "Confidence", "Severity"]

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=100, format="%.3f%%"
            ),
        },
    )
    st.caption("Source/destination IP and port are omitted by design - dropped upstream as "
               "identity/leakage-prone features (see src/data_prep.py).")

    if event.selection and event.selection.get("rows"):
        idx = event.selection["rows"][0]
        record = records[idx]

        @st.dialog(f"Flow #{record['flow_id']}", width="large")
        def flow_detail(record=record):
            severity_label, severity_color = SEVERITY.get(record["predicted_class"], ("UNKNOWN", MUTED))
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                <span style="font-size:1.5rem;font-weight:800;">{record['predicted_class']}</span>
                <span class="badge" style="background-color:{severity_color}18;color:{severity_color};">
                    {severity_label}
                </span>
            </div>
            <div class="mono" style="color:{MUTED};font-size:0.85rem;margin-bottom:18px;">
                confidence {record['confidence']:.3%}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">SHAP Local Explanation</div>', unsafe_allow_html=True)
            shap_df = pd.DataFrame(record["top_shap_features"]).sort_values("shap_value")
            bar_fig = go.Figure(go.Bar(
                x=shap_df["shap_value"], y=shap_df["feature"], orientation="h",
                marker_color=DANGER,
                text=shap_df["value"].apply(lambda v: f"val: {v:g}"),
                textposition="outside",
                textfont={"color": MUTED, "size": 11},
            ))
            bar_fig.update_layout(
                height=240, margin=dict(l=10, r=60, t=10, b=10),
                paper_bgcolor=BG, plot_bgcolor=BG,
                font={"color": TEXT},
                xaxis={"gridcolor": BORDER, "title": "SHAP value"},
                yaxis={"gridcolor": BORDER},
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            st.markdown('<div class="section-label">Grounded LLM Incident Report</div>', unsafe_allow_html=True)
            if st.button("Generate report", type="primary", key=f"gen_{record['flow_id']}"):
                with st.spinner(f"Generating via {backend}..."):
                    try:
                        report = generate_report(record, backend=backend)
                    except Exception as e:
                        report = None
                        st.error(f"Report generation failed: {e}")

                if report:
                    st.markdown(f"**Summary:** {report['summary']}")
                    st.markdown(f"**Evidence:** {report['evidence_explanation']}")
                    if report["mitre_mapping"]:
                        for t in report["mitre_mapping"]:
                            st.markdown(f"- `{t['technique_id']}` {t['technique_name']} ({t['tactic']})")
                    st.markdown(f"**Confidence note:** {report['confidence_note']}")
                    st.markdown("**Recommended actions:**")
                    for a in report["recommended_actions"]:
                        st.markdown(f"- {a}")
            else:
                st.caption("Click 'Generate report' to produce a grounded narrative for this flow.")

        flow_detail()

with tab_insights:
    st.markdown('<div class="section-label">Model Reliability by Class (held-out test set)</div>', unsafe_allow_html=True)
    rel_df = pd.DataFrame(CLASS_RELIABILITY).T.reset_index()
    rel_df.columns = ["Class", "Precision", "Recall", "F1"]
    st.dataframe(
        rel_df, use_container_width=True, hide_index=True,
        column_config={
            "Precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1, format="%.2f"),
            "Recall": st.column_config.ProgressColumn("Recall", min_value=0, max_value=1, format="%.2f"),
            "F1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.2f"),
        },
    )
    st.caption("Bot has the lowest precision (0.63) of any class - see project writeup for the "
               "class-weighting experiment behind this number.")