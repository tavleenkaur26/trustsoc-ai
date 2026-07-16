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

from llm_report import generate_report, CLASS_RELIABILITY  # noqa: E402
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
ACCENT = "#6D5EF7"
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
.stApp {{ background-color: {BG}; }}
[data-testid="stSidebar"] {{
    background-color: #FAFAFC;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
.mono {{ font-family: 'JetBrains Mono', monospace; }}
.eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 0.75rem;
    font-weight: 600;
}}
.page-title {{
    font-size: 2.0rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 2px 0 4px 0;
}}
.page-sub {{ color: {MUTED}; font-size: 0.92rem; margin-bottom: 22px; }}
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
.nav-item {{
    padding: 8px 10px;
    border-radius: 6px;
    font-size: 0.88rem;
    margin-bottom: 2px;
}}

/* Restyle the sidebar nav radio group to look like a real nav menu,
   not default Streamlit radio buttons. */
[data-testid="stSidebar"] div[role="radiogroup"] {{
    gap: 2px;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background-color: transparent;
    border-radius: 8px;
    padding: 9px 12px;
    margin: 0;
    width: 100%;
    transition: background-color 0.12s ease;
    cursor: pointer;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background-color: #EFEDFD;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
    background-color: {ACCENT};
}}
[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {{
    color: white !important;
    font-weight: 600;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{
    display: none;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label p {{
    font-size: 0.92rem;
    color: {TEXT};
}}
/* Tighten default Streamlit vertical spacing in the sidebar */
[data-testid="stSidebar"] .stButton button {{
    border-radius: 8px;
}}
[data-testid="stSidebar"] hr {{
    margin: 14px 0;
}}
/* Compact the default file-uploader drag box for the narrow sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    padding: 10px;
    border-radius: 8px;
    border: 1px dashed {BORDER};
    background-color: {SURFACE};
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span {{
    font-size: 0.78rem;
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {{
    font-size: 0.68rem;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data + session state
# ---------------------------------------------------------------------------
PRED_PATH = ROOT / "schema" / "sample_predictions.json"
with open(PRED_PATH) as f:
    demo_records = json.load(f)

for key, default in [
    ("uploaded_records", None), ("scan_run", False),
    ("last_processed_file", None), ("nav", "Overview"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

records = st.session_state.uploaded_records if st.session_state.uploaded_records is not None else demo_records

if st.session_state.get("upload_msg"):
    st.toast(st.session_state.upload_msg, icon="\u2705")
    del st.session_state.upload_msg

# ---------------------------------------------------------------------------
# Sidebar: brand, nav, controls, live class distribution
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
        <div style="width:32px;height:32px;background:{ACCENT};border-radius:8px;
                    display:flex;align-items:center;justify-content:center;font-size:1rem;">
            \U0001F6E1\uFE0F
        </div>
        <div>
            <div style="font-weight:800;font-size:1.02rem;line-height:1;">
                TrustSOC <span style="color:{ACCENT};">AI</span>
            </div>
            <div class="mono" style="font-size:0.62rem;color:{MUTED};letter-spacing:0.08em;">
                DETECT &middot; EXPLAIN &middot; REPORT
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.nav = st.radio(
        "Navigate", ["Overview", "Detection", "Model Insights", "About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div class="section-label">Run / Upload</div>', unsafe_allow_html=True)
    if st.button("\u25B6 Run demo scan", use_container_width=True):
        st.session_state.uploaded_records = None
        st.session_state.scan_run = True
        st.session_state.last_processed_file = None

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.caption(f"\U0001F4C4 {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB) selected")
        if st.button("\u25B6 Run scan on this file", use_container_width=True, type="primary"):
            with st.spinner("Running model + SHAP..."):
                try:
                    st.session_state.uploaded_records = process_uploaded_csv(uploaded_file)
                    st.session_state.scan_run = True
                    st.session_state.last_processed_file = f"{uploaded_file.name}_{uploaded_file.size}"
                    st.session_state.upload_msg = f"Processed {len(st.session_state.uploaded_records)} flows from upload."
                    st.rerun()
                except ModelNotAvailableError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"Could not process file: {e}")

    st.divider()
    st.markdown('<div class="section-label">LLM Backend</div>', unsafe_allow_html=True)
    backend_label = st.selectbox(
        "LLM backend", ["groq (llama-3.1-8b-instant)", "ollama (llama3.1:8b, local)"],
        label_visibility="collapsed",
    )
    backend = "groq" if backend_label.startswith("groq") else "ollama"
    st.markdown(
        f'<span style="color:{SUCCESS};">\u25CF</span> '
        f'<span class="mono" style="font-size:0.72rem;color:{MUTED};">OPERATIONAL</span>',
        unsafe_allow_html=True,
    )

    if st.session_state.scan_run:
        st.divider()
        st.markdown('<div class="section-label">Class Distribution</div>', unsafe_allow_html=True)
        class_counts = pd.Series([r["predicted_class"] for r in records]).value_counts()
        for cls, count in class_counts.items():
            color = SEVERITY.get(cls, ("", MUTED))[1]
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                f'<span class="mono" style="font-size:0.78rem;">'
                f'<span style="color:{color};">\u25CF</span> {cls}</span>'
                f'<span class="mono" style="font-size:0.78rem;color:{MUTED};">{count}</span>'
                f'</div>', unsafe_allow_html=True,
            )

nav = st.session_state.nav

# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
if nav == "Overview":
    st.markdown('<div class="eyebrow">\u2726 Explainable Intrusion Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Detect threats. Explain every prediction.</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">TrustSOC AI classifies network flows into DDoS, PortScan, Bot, Web Attack, '
        f'and more &mdash; then grounds every incident report in the exact SHAP evidence used by the model. '
        f'No invented IPs. No black-box verdicts.</div>', unsafe_allow_html=True,
    )

    col_pipe, col_status = st.columns([1.4, 1], gap="large")
    with col_pipe:
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

    with col_status:
        if st.session_state.scan_run:
            df = pd.DataFrame(records)
            malicious = df[df["predicted_class"] != "Benign"]
            st.markdown(f"""
            <div class="card">
                <div class="section-label">Session Snapshot</div>
                <div class="stat-value">{len(df)}</div>
                <div class="stat-sub" style="margin-bottom:14px;">flows analyzed</div>
                <div class="stat-value" style="color:{DANGER};">{len(malicious)}</div>
                <div class="stat-sub">flagged malicious &mdash; see the Detection tab</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:30px;color:{MUTED};">
                <div class="mono" style="font-size:0.68rem;letter-spacing:0.1em;margin-bottom:8px;">NO SESSION</div>
                <div style="font-weight:700;color:{TEXT};margin-bottom:6px;">No scan run yet</div>
                <div style="font-size:0.82rem;">Use "Run demo scan" or "Upload CSV" in the sidebar to begin.</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: Detection
# ---------------------------------------------------------------------------
elif nav == "Detection":
    st.markdown('<div class="page-title">Detection</div>', unsafe_allow_html=True)

    if not st.session_state.scan_run:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:50px;color:{MUTED};">
            <div class="mono" style="font-size:0.7rem;letter-spacing:0.1em;margin-bottom:8px;">NO SESSION</div>
            <div style="font-weight:700;color:{TEXT};font-size:1.1rem;margin-bottom:6px;">Run the demo scan to begin</div>
            <div style="font-size:0.85rem;">Use the sidebar to run a demo scan or upload a CSV.</div>
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
    st.markdown('<div class="section-label">Live Detection Results &middot; click a row for SHAP evidence</div>', unsafe_allow_html=True)

    # Filters
    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        severity_filter = st.multiselect(
            "Severity", ["CRITICAL", "SUSPICIOUS", "BENIGN"], default=[],
            placeholder="Filter by severity",
        )
    with fcol2:
        class_filter = st.multiselect(
            "Class", sorted(df["predicted_class"].unique()), default=[],
            placeholder="Filter by predicted class",
        )

    filtered = df.copy()
    filtered["severity"] = filtered["predicted_class"].map(lambda c: SEVERITY.get(c, ("UNKNOWN", MUTED))[0])
    if severity_filter:
        filtered = filtered[filtered["severity"].isin(severity_filter)]
    if class_filter:
        filtered = filtered[filtered["predicted_class"].isin(class_filter)]

    display_df = filtered[["flow_id", "predicted_class", "confidence", "severity"]].copy()
    display_df["confidence"] = display_df["confidence"] * 100
    display_df.columns = ["Flow ID", "Predicted Class", "Confidence", "Severity"]

    st.caption(f"Showing {len(display_df)} of {len(df)} flows")

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
    st.caption("Source/destination IP and port are intentionally omitted from this model - "
           "they are leakage-prone identifiers, not genuine behavioral signals of an attack.")

    if event.selection and event.selection.get("rows"):
        idx = event.selection["rows"][0]
        record = filtered.iloc[idx].to_dict()
        # recover the full record (with top_shap_features) by flow_id
        record = next(r for r in records if r["flow_id"] == record["flow_id"])

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

                    review = report.get("review_assessment", {})
                    if review.get("review_required"):
                        st.error(f"⚠️ Manual review recommended — {review['review_reason']}")
                    else:
                        st.success("✓ Not flagged for mandatory review")

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

# ---------------------------------------------------------------------------
# PAGE: Model Insights
# ---------------------------------------------------------------------------
elif nav == "Model Insights":
    st.markdown('<div class="page-title">Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Per-class reliability from held-out test set evaluation.</div>', unsafe_allow_html=True)
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

# ---------------------------------------------------------------------------
# PAGE: About
# ---------------------------------------------------------------------------
elif nav == "About":
    st.markdown('<div class="page-title">About TrustSOC AI</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="margin-bottom:16px;">
        <div class="section-label">Pipeline</div>
        <div style="color:{MUTED};">Network Traffic &rarr; Feature Extraction &rarr; XGBoost IDS &rarr;
        SHAP (XAI) &rarr; LLM &rarr; SOC Report + MITRE Mapping + Recommended Actions &rarr; Dashboard.</div>
    </div>
    <div class="card" style="margin-bottom:16px;">
        <div class="section-label">Design principle</div>
        <div style="color:{MUTED};">Anything decidable from the data (a threshold comparison, a unit
        conversion, a lookup table) is computed in code, not left to the LLM to derive. The LLM is only
        asked to turn already-verified facts into readable prose &mdash; not to make judgments the code
        can make for it.</div>
    </div>
    <div class="card">
        <div class="section-label">Known limitations</div>
        <div style="color:{MUTED};">MITRE mapping is static per-class rather than contextual per-flow.
        Source/destination IP and port are intentionally omitted (dropped upstream as leakage-prone
        features). A residual hallucination rate remains for genuinely generative claims even after
        code-grounding the decidable parts of the report logic.</div>
    </div>
    """, unsafe_allow_html=True)