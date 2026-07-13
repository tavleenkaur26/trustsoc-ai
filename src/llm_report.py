import json
import sys
from pathlib import Path

SCHEMA_DIR = Path(__file__).parent.parent / "schema"

with open(SCHEMA_DIR / "mitre_mapping.json") as f:
    MITRE_LOOKUP = json.load(f)

with open(SCHEMA_DIR / "class_reliability.json") as f:
    CLASS_RELIABILITY = json.load(f)

CONFIDENCE_THRESHOLD = 0.75
PRECISION_THRESHOLD = 0.80


def compute_review_flag(record: dict) -> dict:
    """
    Deterministically decides whether manual review should be recommended,
    using exactly the same two signals the LLM used to be asked to reason
    about: this flow's confidence, and the predicted class's historical
    precision. Returns a fact for the LLM to state, not a question for it
    to answer.

    Fail-safe direction: if predicted_class isn't in class_reliability.json
    (shouldn't normally happen, but could if the model or schema drifts),
    treat it as precision=0.0, not 1.0. An unknown class must always trigger
    review - silently assuming perfect reliability for something we have no
    data on is the wrong failure direction for a SOC tool.

    Design note: because PRECISION_THRESHOLD (0.80) is set above Bot's
    measured precision (0.63, see class_reliability.json), every Bot
    prediction is flagged for review regardless of this flow's confidence.
    This is intentional, not a bug - Bot's precision ceiling means even a
    99% confident single-flow prediction has a meaningfully elevated
    false-positive risk at the class level, so class-level gating
    overrides per-flow confidence for this class by design.
    """
    confidence = record.get("confidence", 0.0)
    predicted_class = record["predicted_class"]
    class_stats = CLASS_RELIABILITY.get(predicted_class)
    if class_stats is None:
        precision = 0.0  # fail closed: unknown class -> always review
    else:
        precision = class_stats.get("precision", 0.0)

    reasons = []
    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append(f"this flow's confidence ({confidence:.1%}) is below the {CONFIDENCE_THRESHOLD:.0%} threshold")
    if class_stats is None:
        reasons.append(f"'{predicted_class}' is not a recognized class in class_reliability.json, so it cannot be verified as reliable")
    elif precision < PRECISION_THRESHOLD:
        reasons.append(f"the {predicted_class} class has historically lower precision ({precision:.0%}), so false positives are more common for this class")

    return {
        "review_required": bool(reasons),
        "review_reason": "; ".join(reasons) if reasons else
                          "both this flow's confidence and the class's historical precision are high",
    }


SYSTEM_PROMPT = """You are a SOC report-writing assistant. You are given the output of a
machine-learning intrusion detection system for a single network flow:
- predicted attack class and the model's confidence for THIS specific flow
- the top SHAP feature attributions that explain THIS specific prediction
- a static MITRE ATT&CK technique mapping for the predicted class
- the model's HISTORICAL precision/recall/F1 for the predicted class, measured
  on a held-out test set
- review_assessment: a PRECOMPUTED fact (review_required: true/false, and
  review_reason) about whether manual review should be recommended. This has
  already been decided deterministically from the confidence and reliability
  numbers - it is not your judgment to make.

STRICT RULES:
1. Only make claims that are directly supported by the predicted_class, confidence,
   top_shap_features, mitre_techniques, or class_reliability provided to you. Do not
   invent IP addresses, timestamps, user names, hostnames, coordination between hosts,
   attacker intent, or any detail not present in the input.
2. Confidence and class reliability are separate signals - do not conflate them in your
   explanation. A confidence of 0.90+ is high for THIS flow; do not call it "low
   confidence." Separately, if the predicted class has historical precision below 0.80,
   you may mention that false positives are more common for this class.
3. For confidence_note: explicitly state whether review_assessment.review_required is
   true or false (e.g. "manual review is recommended" / "not flagged as required") -
   do not just describe the numbers and leave the reader to infer the status. Paraphrase
   review_assessment.review_reason in your own words. Do NOT re-derive whether review
   is needed, do NOT contradict the given review_required value, and do NOT invent a
   different reason (e.g. a confidence threshold number not given to you). If
   review_required is false, you may still note it's good practice to spot-check, but
   never say review is "unnecessary" - simply state that it is not flagged as required.
4. Explain in plain English what the SHAP features mean for this specific flow rather
   than just repeating numbers. Each feature has TWO separate numbers - do not confuse
   them:
   - "feature_value" is the actual measured reading for this flow (e.g. the real
     packet length, the real duration). This is the number to quote when describing
     what the traffic looked like.
   - "shap_contribution" is how much that feature pushed the model toward this
     prediction - NOT a measurement, NOT a probability, and NOT a "times more likely"
     multiplier. Never quote shap_contribution as if it were the feature_value, and
     never phrase it as "X times more likely." You may say a feature "contributed
     strongly" or "was a minor factor" based on its magnitude, but do not state the
     shap_contribution number as if it were the reading itself.
   The predicted_class tells you what KIND of attack this is - match your description
   to that, not to a generic template:
   - Only describe traffic as a "scan" or "scan probe" if predicted_class is PortScan.
     Never use "scan" or "scan probe" language for DoS/DDoS floods, brute-force
     attempts (FTP-Patator, SSH-Patator), Web Attack, or Bot - these are fundamentally
     different behaviors (sustained flooding, repeated login attempts, or C2
     communication, not reconnaissance) even if their flows are also short.
   - Do not describe the traffic as "coordinated," part of a "campaign," or "waiting
     for instructions" unless a feature directly measures that - flow-level statistics
     alone cannot establish intent. Only call something "an idle period" if its
     feature name is literally "Idle Mean," "Idle Std," "Idle Max," or "Idle Min" -
     for "Flow Duration," "Fwd/Bwd IAT," or "Active" features use neutral language
     like "the flow's duration" or "the time between packets," never "idle period,"
     since those measure different things. A long idle period is a timing fact, not
     evidence of waiting for commands - describe it only as "an extended idle
     period," never attribute intent to it.
   - Before calling a feature value "high," "low," "short," or "long," check the
     actual number given rather than assuming direction from the feature's name.
   - For any feature with a "value_readable" field, quote that string verbatim (or
     trivially reworded) when describing duration/timing in words - never compute or
     relabel the unit yourself, since the raw "value" field is in microseconds and
     manual conversion has repeatedly produced errors.
   - For the "Protocol" feature, if a "protocol_name" field is present, use that name
     exactly - never state a different protocol name from memory (e.g. do not call
     protocol 17 "TCP"; use protocol_name as given).
5. If you are not confident a MITRE technique fits given the evidence, say so instead
   of asserting it flatly.
6. Recommended actions must be generic, standard SOC playbook steps appropriate to the
   attack type - do not claim these are the only options or guaranteed to remediate.
7. Output ONLY valid JSON matching the schema below. No preamble, no markdown fences.

Output JSON schema:
{
  "summary": "1-2 sentence plain-English summary of what was detected",
  "evidence_explanation": "2-4 sentences grounded strictly in the SHAP features given",
  "mitre_mapping": [{"technique_id": "...", "technique_name": "...", "tactic": "..."}],
  "confidence_note": "1-2 sentences: this flow's confidence vs. the class's historical reliability, and the precomputed review_required fact stated in your own words",
  "recommended_actions": ["short imperative action 1", "short imperative action 2", "..."]
}
"""




TIME_FEATURE_MARKERS = ("Duration", "IAT", "Active", "Idle")

# IANA protocol numbers as used by CICFlowMeter. The LLM was repeatedly
# misstating these from memory (e.g. calling protocol 17 "TCP" when it's
# actually UDP) - fixed the same way as duration units: compute the fact
# in code so the LLM only has to state it, not recall it.
PROTOCOL_NAMES = {0: "HOPOPT", 1: "ICMP", 6: "TCP", 17: "UDP"}


def _readable_duration(seconds: float) -> str:
    """Pick the right unit automatically and format it, so the LLM never has to."""
    if seconds >= 1:
        return f"{seconds:.2f} s"
    ms = seconds * 1000
    if ms >= 1:
        return f"{ms:.2f} ms"
    return f"{seconds * 1_000_000:.0f} \u00b5s"


def _annotate_protocol(features: list) -> list:
    annotated = []
    for f in features:
        f = dict(f)
        if f["feature"] == "Protocol":
            try:
                proto_num = int(f["value"])
                f["protocol_name"] = PROTOCOL_NAMES.get(proto_num, f"protocol #{proto_num}")
                f["_note"] = "protocol_name is the correct name for this protocol number - state it exactly as given, do not recall or guess a different protocol name for this number"
            except (TypeError, ValueError):
                pass
        annotated.append(f)
    return annotated


def _annotate_time_features(features: list) -> list:
    """
    CICFlowMeter records duration/inter-arrival/active/idle features in
    microseconds. The LLM was doing unit conversion itself and getting it
    wrong by orders of magnitude, or mislabeling milliseconds as seconds.
    Fix: compute a ready-to-quote, correctly-unit-labeled string in code.
    """
    annotated = []
    for f in features:
        f = dict(f)
        if any(marker in f["feature"] for marker in TIME_FEATURE_MARKERS):
            try:
                seconds = float(f["value"]) / 1_000_000
                f["value_readable"] = _readable_duration(seconds)
                f["_note"] = "value_readable is this same quantity already converted to the right unit - quote it verbatim (or trivially reworded), do not recompute or relabel the unit yourself"
            except (TypeError, ValueError):
                pass
        annotated.append(f)
    return annotated


def _relabel_value_fields(features: list) -> list:
    """
    The LLM was repeatedly quoting shap_value as if it were the actual
    feature reading (e.g. calling a SHAP contribution of 1.73 "the packet
    length minimum" when the real reading was 53.0), and describing SHAP
    values as probability multipliers ("9.1 times more likely"), which is
    not what a SHAP value means. Fix: rename the keys so the two numbers
    are unambiguous, and drop the old ambiguous names entirely.
    """
    relabeled = []
    for f in features:
        f = dict(f)
        new_f = {"feature": f["feature"], "feature_value": f.pop("value"), "shap_contribution": f.pop("shap_value")}
        # carry over any annotations already added (value_readable, protocol_name, _note)
        new_f.update(f)
        relabeled.append(new_f)
    return relabeled


def build_user_prompt(record: dict) -> str:
    predicted_class = record["predicted_class"]
    mitre = MITRE_LOOKUP.get(predicted_class, [])
    reliability = CLASS_RELIABILITY.get(predicted_class, {})
    review_assessment = compute_review_flag(record)
    features = _annotate_time_features(record["top_shap_features"])
    features = _annotate_protocol(features)
    features = _relabel_value_fields(features)
    payload = {
        "flow_id": record["flow_id"],
        "predicted_class": predicted_class,
        "confidence_this_flow": round(record.get("confidence", 0.0), 4),
        "top_shap_features": features,
        "mitre_techniques": mitre,
        "class_historical_reliability": reliability,
        "review_assessment": review_assessment,
    }
    return (
        "Generate the analyst report for this flow. Input evidence:\n\n"
        + json.dumps(payload, indent=2)
    )



def call_local_llm(system_prompt: str, user_prompt: str, model: str = "llama3.1:8b") -> dict:
    """
    Calls a local model via Ollama (free, no API key, runs on your machine).
    Requires `ollama serve` running and the model pulled: `ollama pull llama3.1:8b`.
    Use this for local development/testing only - it will NOT work once deployed
    (Streamlit Cloud has no access to your machine's Ollama instance).
    """
    import requests

    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json()["message"]["content"].strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise ValueError(f"Model did not return parseable JSON:\n{text}")


def call_groq_llm(system_prompt: str, user_prompt: str, model: str = "llama-3.1-8b-instant", max_retries: int = 5) -> dict:
    """
    Calls Groq's hosted API (free tier, fast). Used for deployment, since a
    deployed app has no access to a local Ollama instance.
    Requires GROQ_API_KEY set as an environment variable or in
    .streamlit/secrets.toml (see dashboard/app.py for how it's read).
    Get a free key at https://console.groq.com/keys

    Retries automatically on 429 (rate limit) since the free tier has a
    requests-per-minute cap that's easy to hit when batch-processing many
    flows in a row.
    """
    import os
    import time
    import requests

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys")

    for attempt in range(max_retries):
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 2 ** attempt))
            print(f"  Rate limited, waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(text)

    raise RuntimeError("Groq API rate limit exceeded after max retries.")


def generate_report(record: dict, backend: str = "auto", model: str = None) -> dict:
    """
    backend: "ollama" (local dev), "groq" (deployment), or "auto" (uses Groq if
    GROQ_API_KEY is set in the environment, otherwise falls back to local Ollama).
    """
    import os

    user_prompt = build_user_prompt(record)

    if backend == "auto":
        backend = "groq" if os.environ.get("GROQ_API_KEY") else "ollama"

    if backend == "groq":
        report = call_groq_llm(SYSTEM_PROMPT, user_prompt, model=model or "llama-3.1-8b-instant")
    else:
        report = call_local_llm(SYSTEM_PROMPT, user_prompt, model=model or "llama3.1:8b")

    # Ground-truth fact, computed in code, independent of anything the LLM said.
    # The dashboard can use this directly instead of trusting the LLM's paraphrase.
    report["review_assessment"] = compute_review_flag(record)
    return report


if __name__ == "__main__":
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else "schema/sample_predictions.json"
    with open(path) as f:
        records = json.load(f)

    reports = []
    for i, record in enumerate(records):
        print(f"Generating report for flow {record['flow_id']} ({record['predicted_class']})...")
        report = generate_report(record)
        report["flow_id"] = record["flow_id"]
        reports.append(report)
        print(json.dumps(report, indent=2))
        print("-" * 60)

        if i < len(records) - 1:
            time.sleep(2)  # small proactive pause to avoid tripping the free-tier rate limit

    out_path = Path("outputs/sample_reports.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nSaved {len(reports)} reports to {out_path}")