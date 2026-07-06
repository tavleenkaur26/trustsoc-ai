import shap
import numpy as np
import json

def compute_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_test)

def explain_flow(idx, X_test, y_pred, y_pred_proba, shap_values_full, le, top_k=5):
    pred_class_idx = y_pred[idx]
    pred_class_name = le.classes_[pred_class_idx]
    confidence = float(y_pred_proba[idx, pred_class_idx])

    row_shap = shap_values_full[idx, :, pred_class_idx]
    row_features = X_test.iloc[idx]

    top_indices = np.argsort(np.abs(row_shap))[::-1][:top_k]
    top_features = [
        {"feature": X_test.columns[i], "value": float(row_features.iloc[i]), "shap_value": float(row_shap[i])}
        for i in top_indices
    ]

    return {
        "flow_id": int(idx),
        "predicted_class": pred_class_name,
        "confidence": confidence,
        "top_shap_features": top_features
    }