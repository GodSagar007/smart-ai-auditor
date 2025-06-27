import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, equalized_odds_difference
from sklearn.metrics import accuracy_score

def audit_fairness(y_true, y_pred, sensitive_features: pd.Series):
    """
    Evaluates fairness of predictions across sensitive groups.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Model predictions.
        sensitive_features (pd.Series): Sensitive attribute (e.g., gender, race).

    Returns:
        dict: Fairness metrics and human-readable interpretations.
    """

    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
    }

    frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    # Fairness metrics
    acc_disparity = max(frame.by_group["accuracy"]) - min(frame.by_group["accuracy"])
    sel_disparity = max(frame.by_group["selection_rate"]) - min(frame.by_group["selection_rate"])
    eq_odds_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)

    # Return dictionary
    interpretation = {
        "accuracy_disparity": acc_disparity,
        "selection_rate_disparity": sel_disparity,
        "equalized_odds_difference": eq_odds_diff,
        "is_fair": acc_disparity < 0.1 and sel_disparity < 0.1 and abs(eq_odds_diff) < 0.1,
        "explanations": {
            "accuracy_disparity": "Difference between max and min accuracy across groups. Lower is better (ideal < 0.1).",
            "selection_rate_disparity": "Difference in prediction rates (e.g., approval rates) between groups. Should be < 0.1 for fairness.",
            "equalized_odds_difference": "Measures disparity in false positive/negative rates across groups. Ideal ≈ 0."
        },
        "notes": (
            "✅ Model is likely fair across sensitive groups."
            if acc_disparity < 0.1 and sel_disparity < 0.1 and abs(eq_odds_diff) < 0.1
            else "⚠️ Fairness issues detected. Consider re-training or mitigation."
        )
    }

    return {
        "overall_accuracy": frame.overall["accuracy"],
        "accuracy_by_group": frame.by_group["accuracy"].to_dict(),
        "selection_rate_by_group": frame.by_group["selection_rate"].to_dict(),
        "fairness_interpretation": interpretation
    }
