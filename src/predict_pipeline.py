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
    """Raised when the trained model files haven't been provided yet."""
    pass


def load_model_and_encoder():
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise ModelNotAvailableError(
            f"Trained model not found. Export these two files "
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
    df = df.replace([np.inf, -np.inf], np.nan)
    # Capture original row positions BEFORE dropping any - if dropna() removes
    # rows, the remaining ones must keep their original line number so an
    # analyst can trace a flagged flow_id back to the source CSV. Silently
    # renumbering 0,1,2... after a drop breaks that traceability.
    original_indices = df.index[df.notna().all(axis=1)]
    df = df.dropna()

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
    for i, orig_idx in enumerate(original_indices):
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
            "flow_id": int(orig_idx),  # original CSV row number, not a renumbered position
            "predicted_class": label_encoder.classes_[pred_idx],
            "confidence": confidence,
            "top_shap_features": [
                {"feature": f, "value": float(v), "shap_value": float(s)}
                for f, s, v in top_features
            ],
        })
    return records