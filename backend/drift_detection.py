# backend/drift_detection.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, entropy

def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, threshold: float = 0.1):
    """
    Detects data drift between reference and current dataset using KS test and Jensen-Shannon divergence.

    Args:
        reference_data (pd.DataFrame): Original training data.
        current_data (pd.DataFrame): Incoming data to compare.
        threshold (float): Threshold for flagging drift (default = 0.1).

    Returns:
        dict: Drift scores, interpretation, and per-feature stats.
    """
    drift_report = {
        "drift_detected": False,
        "feature_stats": {},
        "drift_flags": {},
        "drift_score": 0,
        "notes": ""
    }

    total_drifted = 0
    num_features = 0

    for col in reference_data.columns:
        if col not in current_data.columns:
            continue
        if reference_data[col].dtype not in [np.float64, np.int64, np.int32, np.float32]:
            continue  # skip categorical

        ref = reference_data[col].dropna()
        curr = current_data[col].dropna()

        if len(ref) < 10 or len(curr) < 10:
            continue

        num_features += 1
        ks_stat, ks_pval = ks_2samp(ref, curr)
        js_div = entropy(
            np.histogram(ref, bins=20, density=True)[0] + 1e-9,
            np.histogram(curr, bins=20, density=True)[0] + 1e-9,
            base=2
        )

        drifted = ks_pval < 0.05 or js_div > threshold
        drift_report["feature_stats"][col] = {
            "ks_p_value": ks_pval,
            "js_divergence": js_div,
            "drifted": drifted
        }
        drift_report["drift_flags"][col] = drifted

        if drifted:
            total_drifted += 1

    if num_features > 0:
        drift_score = total_drifted / num_features
    else:
        drift_score = 0

    drift_report["drift_score"] = drift_score
    drift_report["drift_detected"] = drift_score > threshold
    drift_report["notes"] = (
        "⚠️ Drift detected in input features. Consider retraining."
        if drift_report["drift_detected"]
        else "✅ No significant drift detected."
    )

    return drift_report
