# TrustSOC AI

**Trustworthy AI Security Operations Center (SOC) Analyst Copilot**

TrustSOC AI is an explainable Intrusion Detection System (IDS) that combines machine learning, SHAP explainability, and Large Language Models (LLMs) to generate trustworthy incident reports. Instead of presenting analysts with only an attack label, the system explains why a prediction was made and converts verified evidence into an easy-to-understand security report.

Unlike conventional LLM-based SOC assistants, TrustSOC AI grounds every generated explanation in SHAP feature attributions and pre-computed security information, reducing the risk of unsupported or hallucinated outputs.

---

## Motivation

Machine learning-based intrusion detection systems have achieved high detection accuracy but often operate as black boxes, making it difficult for Security Operations Center (SOC) analysts to understand why a network flow was classified as malicious.

Recent work has attempted to improve interpretability by integrating Large Language Models (LLMs) into security workflows to generate incident summaries. While these reports are more readable, they may include hallucinations or unsupported statements that are not grounded in the underlying model prediction.

TrustSOC AI addresses this challenge by combining:

- Accurate intrusion detection
- Model-level explainability using SHAP
- Grounded LLM-generated incident reports
- MITRE ATT&CK mapping
- Actionable mitigation recommendations

---

## Features

- 11-class intrusion detection using XGBoost
- SHAP-based explanations for every prediction
- Grounded LLM-generated incident reports
- MITRE ATT&CK technique mapping
- Risk scoring
- Analyst-friendly Streamlit dashboard
- Explainable predictions instead of black-box outputs

---

## System Architecture

```
Network Traffic
      │
      ▼
Feature Extraction
      │
      ▼
XGBoost Intrusion Detection Model
      │
      ▼
SHAP Feature Attribution
      │
      ▼
Verified Context Builder
      │
      ▼
Grounded LLM Report Generator
      │
      ▼
MITRE ATT&CK Mapping
      │
      ▼
Streamlit Dashboard
```

---

## Dataset

The project uses the CICIDS2017 dataset proposed by Sharafaldin et al. (2018).

To improve reproducibility, experiments were performed using the cleaned Kaggle mirror: **dhoogla/cicids2017**

**Attack categories included:**
- Benign
- Bot
- DDoS
- DoS GoldenEye
- DoS Hulk
- DoS Slowhttptest
- DoS Slowloris
- FTP-Patator
- PortScan
- SSH-Patator
- Web Attack (merged)

**Classes excluded due to insufficient samples:**
- Heartbleed
- Infiltration

---

## Model

| Component | Detail |
|---|---|
| Algorithm | XGBoost Classifier |
| Explainability | SHAP (TreeSHAP) |
| Report Generation | Llama 3 via Groq API |
| Dashboard | Streamlit |

---

## Performance

| Metric | Score |
|---|---|
| Accuracy | 0.9990 |
| Macro Precision | 0.9485 |
| Macro Recall | 0.9824 |
| Macro F1 | 0.9626 |
| Weighted F1 | 0.9990 |

Evaluated on a held-out test set of 462,753 samples across 11 classes.

The **Bot class remains the most challenging category** because of class imbalance and overlapping traffic characteristics — precision for this class (0.6276) is notably lower than all other classes, despite strong recall (0.9338), reflecting a higher false-positive rate driven by limited training samples (287 in the test set) and overlap with other traffic patterns.

<details>
<summary>Full per-class classification report</summary>

```
              precision    recall  f1-score   support
           0     0.9998    0.9991    0.9994    395464
           1     0.6276    0.9338    0.7507       287
           2     0.9999    0.9996    0.9998     25603
           3     0.9956    0.9966    0.9961      2057
           4     0.9980    0.9993    0.9987     34569
           5     0.9184    0.9904    0.9531      1046
           6     0.9935    0.9889    0.9912      1077
           7     0.9950    0.9983    0.9966      1186
           8     0.9237    0.9284    0.9260       391
           9     0.9984    0.9953    0.9969       644
          10     0.9836    0.9767    0.9801       429

    accuracy                         0.9990    462753
   macro avg     0.9485    0.9824    0.9626    462753
weighted avg     0.9991    0.9990    0.9990    462753
```

</details>

---

## Example Workflow

1. Network flow is provided to the IDS.
2. XGBoost predicts the attack category.
3. SHAP identifies the features contributing most to the prediction.
4. Verified statistics are passed to the LLM.
5. The LLM generates an incident report using only grounded information.
6. MITRE ATT&CK mappings and mitigation recommendations are displayed.
7. Results are presented through the Streamlit dashboard.

---

## Project Structure

```
trustsoc-ai/
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── explain.py
│   └── report_generator.py
│
├── outputs/
│
├── models/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/tavleenkaur26/trustsoc-ai.git
cd trustsoc-ai
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Running the Project

**Train the model**
```bash
python src/train.py
```

**Launch the dashboard**
```bash
streamlit run dashboard/app.py
```

Configure your Groq API key using either:
- `.streamlit/secrets.toml`, or
- `GROQ_API_KEY=<your_key>`

---

## Design Philosophy

TrustSOC AI follows a simple principle:

> Anything that can be computed deterministically should be computed in code, not generated by an LLM.

The LLM is used only to convert verified evidence into a human-readable report. All attack predictions, SHAP explanations, confidence values, risk scores, and MITRE ATT&CK mappings are produced before the prompt is sent to the language model.

---

## Limitations

- Results depend on the quality of the CICIDS2017 dataset.
- Certain attack classes contain very few samples.
- The Bot class remains comparatively difficult to classify.
- MITRE ATT&CK mappings are static and class-based.
- SHAP explanations describe the model's decision rather than establishing causality.

---

## Future Work

- Real-time packet capture integration
- Multi-model ensemble IDS
- Support for additional network datasets
- Retrieval-Augmented Generation (RAG) for threat intelligence
- Dynamic MITRE ATT&CK mapping
- Confidence calibration and uncertainty estimation
- Integration with SIEM platforms
- Analyst feedback loop for continual learning

---

## Authors

**Tavleen Kaur** (20501012025)
- B.Tech Computer Science Engineering
- Indira Gandhi Delhi Technical University for Women (IGDTUW)

**Tanisha Sharma** (20201012025)
- B.Tech Computer Science Engineering
- Indira Gandhi Delhi Technical University for Women (IGDTUW)
