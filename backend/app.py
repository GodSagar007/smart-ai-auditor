from backend.loader import load_model, load_data
from backend.bias_audit import audit_fairness

# Keywords to auto-detect sensitive attributes
SENSITIVE_KEYWORDS = ['gender', 'sex', 'race', 'ethnicity']

def detect_sensitive_columns(columns):
    return [col for col in columns if any(keyword in col.lower() for keyword in SENSITIVE_KEYWORDS)]

def main():
    print("🔍 Smart AI Auditor - Interactive Mode")

    # Load model and data
    model = load_model()
    df = load_data()

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

if __name__ == "__main__":
    main()
