import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.loader import load_model, load_data, load_reference_data
from backend.bias_audit import audit_fairness
from backend.explainability import explain_with_shap, explain_with_lime
from backend.drift_detection import detect_data_drift

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Keywords to auto-detect sensitive attributes
SENSITIVE_KEYWORDS = ['gender', 'sex', 'race', 'ethnicity']

def detect_sensitive_columns(columns):
    return [col for col in columns if any(keyword in col.lower() for keyword in SENSITIVE_KEYWORDS)]

def main():
    print("🔍 Smart AI Auditor - Interactive Mode")

    # Load model and data
    model = load_model()
    df = load_data()
    reference_df = load_reference_data()

    print(f"\n📊 Available columns in your data: {list(df.columns)}\n")

    # Ask for target column
    target_col = input("🎯 Enter the name of the target column (label): ").strip()
    if target_col not in df.columns:
        raise ValueError(f"❌ Column '{target_col}' not found in data.")

    # Ask for sensitive columns or auto-detect
    sensitive_input = input("🧪 Enter sensitive column(s) (comma-separated) or leave blank for auto-detect: ").strip()
    if sensitive_input:
        sensitive_cols = [col.strip() for col in sensitive_input.split(",")]
        for col in sensitive_cols:
            if col not in df.columns:
                raise ValueError(f"❌ Sensitive column '{col}' not found in data.")
    else:
        sensitive_cols = detect_sensitive_columns(df.columns)
        if sensitive_cols:
            print(f"✅ Auto-detected sensitive columns: {sensitive_cols}")
        else:
            print("⚠️ No sensitive columns detected. Skipping bias audit.")
            sensitive_cols = []

    # Prepare features and target
    X = df.drop(columns=[target_col])
    y_true = df[target_col]
    y_pred = model.predict(X)

    print("✅ Model prediction complete.\n")

    # Run audit per sensitive feature
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

    if not sensitive_cols:
        print("🚫 No sensitive features audited. Done.")

    # === Drift Detection ===
    if reference_df is not None:
        print("\n📈 Running Drift Detection...")
        drift_results = detect_data_drift(reference_df, df)

        print("📊 Drift Summary:")
        for feature, drifted in drift_results["drift_flags"].items():
            status = "⚠️ Drifted" if drifted else "✅ Stable"
            print(f"• {feature}: {status}")

        print("\n📘 Drift Detection uses statistical tests (e.g., KS Test) to detect if feature distributions have shifted.")
        print(f"🔎 Drift Detected in {sum(drift_results['drift_flags'].values())}/{len(drift_results['drift_flags'])} features.")
    else:
        print("\n🚫 Drift detection skipped (no reference_data.csv found).")

    # === Explainability ===
    print("\n🧠 Explainability (SHAP & LIME)")
    print("\n📌 Explaining prediction for first instance:")
    sample = X.iloc[[0]]

    # SHAP
    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
        print("\n🔎 SHAP Explanation for Sample #1:")
        try:
            explain_with_shap(model, sample)
        except Exception as e:
            print(f"⚠️ SHAP explanation failed: {e}")
    else:
        print("⚠️ SHAP not applicable: Model is not tree-based (RandomForest, GradientBoosting, etc.)")

    # LIME
    if isinstance(model, (LogisticRegression, RandomForestClassifier)):
        print("\n🔍 LIME Explanation for Sample #1:")
        try:
            explain_with_lime(model, X, sample)
        except Exception as e:
            print(f"⚠️ LIME explanation failed: {e}")
    else:
        print("⚠️ LIME not applicable: Only implemented for LogisticRegression or tree-based classifiers")

if __name__ == "__main__":
    main()
