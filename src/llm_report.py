import json
import sys
from pathlib import Path

SCHEMA_DIR = Path(__file__).parent.parent / "schema"

with open(SCHEMA_DIR / "mitre_mapping.json") as f:
    MITRE_LOOKUP = json.load(f)

with open(SCHEMA_DIR / "class_reliability.json") as f:
    CLASS_RELIABILITY = json.load(f)


SYSTEM_PROMPT = """You are a SOC report-writing assistant. You are given the output of a
machine-learning intrusion detection system for a single network flow:
- predicted attack class and the model's confidence for THIS specific flow
- the top SHAP feature attributions that explain THIS specific prediction
- a static MITRE ATT&CK technique mapping for the predicted class
- the model's HISTORICAL precision/recall/F1 for the predicted class, measured
  on a held-out test set (this tells you how trustworthy this class has been
  in general, which is a separate fact from this flow's confidence score)

STRICT RULES:
1. Only make claims that are directly supported by the predicted_class, confidence,
   top_shap_features, mitre_techniques, or class_reliability provided to you. Do not
   invent IP addresses, timestamps, user names, hostnames, coordination between hosts,
   attacker intent, or any detail not present in the input.
2. Confidence and class reliability are separate signals - do not conflate them.
   A confidence of 0.90+ is high for THIS flow; do not call it "low confidence."
   Separately, if the predicted class has historical precision below 0.80, say so
   explicitly and note that false positives are more common for this class, even if
   this flow's own confidence is high.
3. Manual review guidance is tied ONLY to this flow's confidence, never to how high
   it is. If confidence is below 0.75, say the prediction is lower-confidence and an
   analyst should manually verify. If confidence is 0.75 or above, do not say review
   is needed "because" confidence is high - that reverses the rule. High confidence on
   its own is never a reason to recommend review; low confidence or low class
   reliability (rule 2) are the only two valid reasons.
3.5. Never state or imply that manual review is unnecessary or can be skipped,
   regardless of how high the confidence or class reliability is. Even a
   perfect-precision class can have an unseen edge case. You may say
   "high confidence, low false-positive risk" but never "review not needed"
   or similar. The decision to skip review belongs to the analyst, not you.
4. Explain in plain English what the SHAP features mean for this specific flow rather
   than just repeating numbers. The predicted_class tells you what KIND of attack this
   is - match your description to that, not to a generic template:
   - Only describe traffic as a "scan" or "scan probe" if predicted_class is PortScan.
     Never use "scan" or "scan probe" language for DoS/DDoS floods, brute-force
     attempts (FTP-Patator, SSH-Patator), Web Attack, or Bot - these are fundamentally
     different behaviors (sustained flooding, repeated login attempts, or C2
     communication, not reconnaissance) even if their flows are also short.
   - Do not describe the traffic as "coordinated," part of a "campaign," or "waiting
     for instructions" unless a feature directly measures that - flow-level statistics
     alone cannot establish intent.
   - Before calling a feature value "high," "low," "short," or "long," check the
     actual number given rather than assuming direction from the feature's name.
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
  "confidence_note": "1-2 sentences distinguishing this flow's confidence from the class's historical reliability, and whether manual review is advised",
  "recommended_actions": ["short imperative action 1", "short imperative action 2", "..."]
}
"""


def build_user_prompt(record: dict) -> str:
    predicted_class = record["predicted_class"]
    mitre = MITRE_LOOKUP.get(predicted_class, [])
    reliability = CLASS_RELIABILITY.get(predicted_class, {})
    payload = {
        "flow_id": record["flow_id"],
        "predicted_class": predicted_class,
        "confidence_this_flow": record.get("confidence"),
        "top_shap_features": record["top_shap_features"],
        "mitre_techniques": mitre,
        "class_historical_reliability": reliability,
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
        return call_groq_llm(SYSTEM_PROMPT, user_prompt, model=model or "llama-3.1-8b-instant")
    return call_local_llm(SYSTEM_PROMPT, user_prompt, model=model or "llama3.1:8b")


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