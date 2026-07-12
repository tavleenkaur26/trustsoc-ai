# TrustSOC AI — Trustworthy AI SOC Analyst Copilot

A network intrusion detection pipeline that pairs an explainable ML classifier
with a grounded LLM report generator, so SOC analysts get attack detection
that shows its work instead of a black-box verdict.

## Problem

Traditional ML-based IDS are accurate but opaque. LLM-based SOC assistants
are readable but prone to hallucination. TrustSOC AI grounds every generated
report strictly in SHAP evidence and pre-computed facts, so the LLM narrates
verified information rather than inventing it.

## Pipeline

Network Traffic (CICIDS2017 schema)
→ Feature Extraction
→ XGBoost IDS (11-class)
→ SHAP explainability (per-flow, per-class)
→ LLM report generation (Groq/Llama, grounded in SHAP + reliability data)
→ MITRE ATT&CK mapping (static, per-class)
→ Streamlit dashboard

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Sharafaldin et
al., 2018), sourced via a cleaned Kaggle mirror
([dhoogla/cicids2017](https://www.kaggle.com/datasets/dhoogla/cicids2017))
for reproducibility within project time constraints. Known dataset-quality
issues (label noise, flow-construction bugs) are documented in Engelen et
al. (2021); see Limitations below for how this affected our methodology.

11 classes used: Benign, Bot, DDoS, DoS GoldenEye, DoS Hulk, DoS
Slowhttptest, DoS slowloris, FTP-Patator, PortScan, SSH-Patator, Web Attack
(merged from 3 sub-types). Heartbleed (n=11) and Infiltration (n=36) were
excluded — insufficient samples for meaningful train/test evaluation.

## Model performance

| Metric | Value |
|---|---|
| Macro F1 | 0.96 |
| Weighted F1 | 1.00 |
| Weakest class | Bot (precision 0.63, recall 0.93, F1 0.75) |

Full per-class report in [outputs/](outputs/). See Limitations for discussion
of the Bot class and dataset separability.

## Setup

```bash
git clone https://github.com/tavleenkaur26/trustsoc-ai
cd trustsoc-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the dataset (see Dataset section) into `data/raw/` as `.parquet`
files. Then:

```bash
python src/train.py          # trains model, saves to outputs/
streamlit run dashboard/app.py
```

Requires a free [Groq API key](https://console.groq.com/keys) set in
`.streamlit/secrets.toml` or as `GROQ_API_KEY` env var for LLM report
generation.

## Design principle

Anything decidable from data (a threshold comparison, a unit conversion, a
lookup table) is computed in code, not left to the LLM to derive. The LLM's
role is restricted to turning already-verified facts into readable prose.

## Team
Tanisha Sharma (20201012025) - IGDTUW
Tavleen Kaur (20501012025) - IGDTUW
