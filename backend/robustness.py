# backend/robustness.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_robustness(model, X: pd.DataFrame, y_true: pd.Series, noise_level: float = 0.05):
    """
    Evaluates model robustness by injecting noise into numerical features and
    measuring the drop in accuracy.

    Args:
        model: Trained ML model with predict method
        X (pd.DataFrame): Original feature set
        y_true (pd.Series): True labels
        noise_level (float): Percentage of noise to inject into numerical features

    Returns:
        dict: Original accuracy, noisy accuracy, drop, and interpretation
    """
    X_noisy = X.copy()
    numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return {
            "error": "No numeric columns found to evaluate robustness."
        }

    # Add Gaussian noise to numeric columns
    for col in numeric_cols:
        std_dev = X_noisy[col].std()
        if std_dev > 0:
            noise = np.random.normal(loc=0, scale=noise_level * std_dev, size=len(X_noisy))
            X_noisy[col] += noise

    y_pred_orig = model.predict(X)
    y_pred_noisy = model.predict(X_noisy)

    acc_orig = accuracy_score(y_true, y_pred_orig)
    acc_noisy = accuracy_score(y_true, y_pred_noisy)
    acc_drop = acc_orig - acc_noisy

    interpretation = (
        "✅ Model is robust to slight perturbations."
        if acc_drop < 0.05 else
        "⚠️ Model performance drops significantly under noise. Consider regularization or adversarial training."
    )

    return {
        "original_accuracy": acc_orig,
        "noisy_accuracy": acc_noisy,
        "accuracy_drop": acc_drop,
        "notes": interpretation
    }
