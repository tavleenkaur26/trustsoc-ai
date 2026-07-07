"""
src/predict_pipeline.py

Turns a raw CICIDS-format CSV upload into the same JSON schema used
everywhere else in this project (schema/prediction_output.json), by running
Person 1's trained model + SHAP TreeExplainer on it.

REQUIRES two model artifacts that Person 1 needs to export (see the
joblib.dump() calls already in their training notebook) and place here:
    models/final_xgb_model.pkl   - the trained XGBoost classifier
    models/label_encoder.pkl     - the LabelEncoder used on y during training

Until those files exist, CSV upload raises a clear error explaining what's
missing. It will NOT fabricate predictions to fill the gap.
"""

import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "outputs"
MODEL_PATH = MODELS_DIR / "final_xgb_model.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# Same leakage-prone / non-feature columns dropped during training
# (see src/data_prep.py - keep this list in sync with theirs).
DROP_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "source_file", "Label"]


class ModelNotAvailableError(Exception):
    """Raised when Person 1's trained model files haven't been provided yet."""
    pass


def load_model_and_encoder():
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise ModelNotAvailableError(
            f"Trained model not found. Ask Person 1 to export these two files "
            f"(joblib.dump) and place them in the models/ folder:\n"
            f"  - {MODEL_PATH.name}\n"
            f"  - {ENCODER_PATH.name}"
        )
    import joblib
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder


def process_uploaded_csv(file, top_k: int = 5) -> list[dict]:
    """
    file: a file-like object (e.g. from st.file_uploader) containing a raw
    CICIDS-format CSV (same shape as the training data).

    Returns a list of records matching schema/prediction_output.json:
    [{flow_id, predicted_class, confidence, top_shap_features: [...]}, ...]
    """
    import shap

    model, label_encoder = load_model_and_encoder()

    df = pd.read_csv(file, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        raise ValueError("No valid rows remain after cleaning (check for NaN/Inf-heavy input).")

    expected_cols = list(getattr(model, "feature_names_in_", df.columns))
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded CSV is missing {len(missing)} column(s) the model expects, "
            f"e.g. {missing[:5]}. Make sure the CSV matches the training schema."
        )
    X = df[expected_cols]

    preds = model.predict(X)
    probas = model.predict_proba(X)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    records = []
    for i in range(len(X)):
        pred_idx = int(preds[i])
        confidence = float(probas[i][pred_idx])

        # shap_values shape differs by SHAP version: list-per-class or a
        # single (n_samples, n_features, n_classes) array - handle both.
        if isinstance(shap_values, list):
            row_shap = shap_values[pred_idx][i]
        else:
            row_shap = shap_values[i, :, pred_idx]

        top_features = sorted(
            zip(X.columns, row_shap, X.iloc[i].values),
            key=lambda t: abs(t[1]), reverse=True,
        )[:top_k]

        records.append({
            "flow_id": i,
            "predicted_class": label_encoder.classes_[pred_idx],
            "confidence": confidence,
            "top_shap_features": [
                {"feature": f, "value": float(v), "shap_value": float(s)}
                for f, s, v in top_features
            ],
        })
    return records