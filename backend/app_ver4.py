# backend/app_ver4.py
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.loader import load_model, load_data
from backend.bias_audit import audit_fairness
from backend.explainability import explain_with_shap, explain_with_lime
from backend.drift_detection import detect_data_drift
from backend.robustness import evaluate_robustness

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

SENSITIVE_KEYWORDS = ['gender', 'sex', 'race', 'ethnicity']

def detect_sensitive_columns(columns):
    return [col for col in columns if any(keyword in col.lower() for keyword in SENSITIVE_KEYWORDS)]

def main():
    print("🔍 Smart AI Auditor - Interactive Mode")

    model = load_model()
    df = load_data()

    print(f"\n📊 Available columns in your data: {list(df.columns)}\n")

    target_col = input("🎯 Enter the name of the target column (label): ").strip()
    if target_col not in df.columns:
        raise ValueError(f"❌ Column '{target_col}' not found in data.")

    sensitive_input = input("🧪 Enter sensitive column(s) (comma-separated) or leave blank for auto-detect: ").strip()
    if sensitive_input:
        sensitive_cols = [col.strip() for col in sensitive_input.split(",")]
        for col in sensitive_cols:
            if col not in df.columns:
                raise ValueError(f"❌ Sensitive column '{col}' not found in data.")
    else:
        sensitive_cols = detect_sensitive_columns(df.columns)
        print(f"✅ Auto-detected sensitive columns: {sensitive_cols}" if sensitive_cols else "⚠️ No sensitive columns detected.")
    
    X = df.drop(columns=[target_col])
    y_true = df[target_col]
    y_pred = model.predict(X)

    print("✅ Model prediction complete.\n")

    # Fairness
    for col in sensitive_cols:
        print(f"\n📋 Running fairness audit for: '{col}'")
        results = audit_fairness(y_true, y_pred, sensitive_features=X[col])
        interp = results["fairness_interpretation"]

        print("\n📊 Fairness Metrics")
        print(f"▪ Overall Accuracy              : {results['overall_accuracy']:.4f}")
        print(f"▪ Accuracy by Group            : {results['accuracy_by_group']}")
        print(f"▪ Selection Rate by Group      : {results['selection_rate_by_group']}")
        print(f"▪ Accuracy Disparity           : {interp['accuracy_disparity']:.4f}")
        print(f"▪ Selection Rate Disparity     : {interp['selection_rate_disparity']:.4f}")
        print(f"▪ Equalized Odds Difference    : {interp['equalized_odds_difference']:.4f}")

        print("\n📘 Metric Definitions:")
        print("• Accuracy Disparity: Difference in accuracy between best and worst performing groups. Ideal < 0.1")
        print("• Selection Rate Disparity: Difference in selection (positive prediction) rates across groups. Ideal < 0.1")
        print("• Equalized Odds Difference: Measures difference in error rates (false pos/neg) between groups. Ideal ≈ 0")

        print(f"\n🔎 Final Verdict: {interp['notes']}\n")

    # Explainability
    print("\n🧠 Explainability (SHAP & LIME)")
    sample = X.iloc[[0]]
    print("\n📌 Explaining prediction for first instance:")

    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
        print("\n🔎 SHAP Explanation for Sample #1:")
        explain_with_shap(model, sample)
    else:
        print("⚠️ SHAP not applicable: Model is not tree-based.")

    if isinstance(model, (LogisticRegression, RandomForestClassifier)):
        print("\n🔍 LIME Explanation for Sample #1:")
        explain_with_lime(model, X, sample)
    else:
        print("⚠️ LIME not applicable: Only supported for LogisticRegression or RandomForest.")

    # Drift Detection
    print("\n📦 Data Drift Detection")
    ref_path = input("📁 Enter path to reference data CSV (e.g., 'data/reference_data.csv'): ").strip()
    if not os.path.exists(ref_path):
        print("❌ Reference data path not found. Skipping drift check.")
    else:
        reference_data = pd.read_csv(ref_path)
        drift_results = detect_data_drift(reference_data, X)

        print("\n📊 Drift Summary:")
        print(f"▪ Drift Score        : {drift_results['drift_score']:.4f}")
        print(f"▪ Drift Detected     : {drift_results['drift_detected']}")
        print(f"▪ Notes              : {drift_results['notes']}")

        for feature, stats in drift_results["feature_stats"].items():
            print(f"  • {feature}: Drifted={stats['drifted']}, KS_p={stats['ks_p_value']:.4f}, JS Divergence={stats['js_divergence']:.4f}")

    # Robustness
    print("\n🧪 Model Robustness Evaluation")
    robustness_results = evaluate_robustness(model, X, y_true)

    if "error" in robustness_results:
        print(f"❌ Robustness check skipped: {robustness_results['error']}")
    else:
        print(f"▪ Original Accuracy     : {robustness_results['original_accuracy']:.4f}")
        print(f"▪ Noisy Accuracy        : {robustness_results['noisy_accuracy']:.4f}")
        print(f"▪ Accuracy Drop         : {robustness_results['accuracy_drop']:.4f}")
        print(f"▪ Notes                 : {robustness_results['notes']}")

if __name__ == "__main__":
    main()
