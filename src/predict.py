"""
src/predict.py
--------------
Inference module — loads trained model and returns risk scores + SHAP values.
Used by the Streamlit app.
"""

import pickle
import numpy as np
import pandas as pd
import shap

MODEL_PATH = 'models/xgb_final.pkl'

_model    = None
_explainer = None


def _load_model():
    global _model, _explainer
    if _model is None:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer


def predict_risk(input_df: pd.DataFrame) -> dict:
    """
    Parameters
    ----------
    input_df : Single-row DataFrame with engineered features

    Returns
    -------
    dict with:
        probability  : float  — default probability (0-1)
        risk_label   : str    — 'Low' / 'Medium' / 'High'
        shap_values  : array  — SHAP values for waterfall plot
        base_value   : float  — SHAP expected value
        feature_names: list
    """
    model, explainer = _load_model()

    prob = model.predict_proba(input_df)[0][1]

    # Risk tiers (tuned for ~8% base rate)
    if prob < 0.15:
        label = 'Low'
    elif prob < 0.35:
        label = 'Medium'
    else:
        label = 'High'

    # SHAP explanation
    shap_vals = explainer(input_df)

    return {
        'probability'  : round(float(prob), 4),
        'risk_label'   : label,
        'shap_values'  : shap_vals,
        'feature_names': input_df.columns.tolist(),
    }


def risk_color(label: str) -> str:
    """Returns a hex color for a risk label."""
    return {'Low': '#2ECC71', 'Medium': '#F39C12', 'High': '#E74C3C'}.get(label, '#888')
