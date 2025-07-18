# frontend/app.py
import os, sys, pickle, joblib
import streamlit as st
import pandas as pd

# ——— Ensure we can import backend modules (assumes ../backend exists) ———
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import cloudpickle
except ImportError:
    cloudpickle = None  # optional

# ——— Backend audit modules ———
from backend.bias_audit import audit_fairness
from backend.drift_detection import detect_data_drift
from backend.robustness import evaluate_robustness
from backend.explainability import explain_with_shap_ver2

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Smart AI Auditor", layout="wide")
st.title("🧠 Smart AI Auditor")

# ----------------- Sidebar: Uploads -----------------
st.sidebar.header("Upload Model & Data")
model_file = st.sidebar.file_uploader("Model (.pkl)", type=["pkl"])
data_file  = st.sidebar.file_uploader("Dataset (.csv)", type=["csv"])
label_col  = st.sidebar.text_input("Label Column (ground truth)", value="label")

# ----------------- Model Loader -----------------
def load_model(f):
    loaders = [
        ("joblib",       joblib.load),
        ("pickle",       pickle.load),
        ("cloudpickle",  cloudpickle.load if cloudpickle else None)
    ]
    for name, loader in loaders:
        if loader is None:
            continue
        try:
            f.seek(0)
            return loader(f)
        except Exception as e:
            st.warning(f"• { name } loader failed: { e }")
    return None

# ----------------- Main Logic -----------------
if model_file and data_file:
    model = load_model(model_file)
    if model is None:
        st.error("❌ Could not load model. Make sure it was saved with joblib / pickle / cloudpickle.")
        st.stop()

    df = pd.read_csv(data_file)
    if label_col not in df.columns:
        st.error(f"❌ Label column '{label_col}' not in dataset.")
        st.stop()

    X, y = df.drop(columns=[label_col]), df[label_col]
    st.success("✅ Model & data loaded!")

    # ----------------- Select Audit -----------------
    audit_choice = st.selectbox("Choose an audit to run", 
                                ["Bias / Fairness", "Explainability (SHAP)", "Data Drift", "Robustness"])

    # -------------- Bias / Fairness -----------------
    if audit_choice == "Bias / Fairness":
        st.subheader("⚖️ Bias & Fairness Audit")
        sensitive_cols = st.sidebar.multiselect("Sensitive attribute(s)", options=X.columns.tolist())
        if not sensitive_cols:
            st.warning("Select at least one sensitive column in the sidebar.")
        else:
            y_pred = model.predict(X)
            for col in sensitive_cols:
                st.markdown(f"### 🔍 Results for `{col}`")
                result = audit_fairness(y_true=y, y_pred=y_pred, sensitive_features=X[col])

                st.write("**Accuracy by Group**", result["accuracy_by_group"])
                st.write("**Selection Rate by Group**", result["selection_rate_by_group"])
                st.markdown(f"**Verdict:** { result['fairness_interpretation']['notes'] }")
                with st.expander("Details"):
                    st.json(result["fairness_interpretation"])
                st.divider()

    # -------------- Explainability (SHAP) -----------
    elif audit_choice == "Explainability (SHAP)":
        st.subheader("🧠 SHAP Explainability")
        sample = X.sample(n=min(5, len(X)), random_state=42)
        explanation = explain_with_shap_ver2(model, sample)
        if "figures" in explanation:
            for fig in explanation["figures"]:
                st.pyplot(fig)
        else:
            st.error(explanation.get("error", "SHAP explanation failed."))

    # -------------- Data Drift -----------------------
    elif audit_choice == "Data Drift":
        st.subheader("📉 Data Drift Detection")
        ref = X.sample(frac=0.5, random_state=1)
        cur = X.sample(frac=0.5, random_state=2)
        report = detect_data_drift(ref, cur)
        st.metric("Drift Score", f"{report['drift_score']:.2f}")
        st.write(report["notes"])
        with st.expander("Feature‑level stats"):
            st.json(report["feature_stats"])

    # -------------- Robustness -----------------------
    elif audit_choice == "Robustness":
        st.subheader("🛡️ Robustness Test")
        res = evaluate_robustness(model, X, y)
        if "error" in res:
            st.error(res["error"])
        else:
            st.metric("Original Accuracy", f"{res['original_accuracy']:.3f}")
            st.metric("Noisy Accuracy",    f"{res['noisy_accuracy']:.3f}")
            st.metric("Accuracy Drop",     f"{res['accuracy_drop']:.3f}")
            st.write(res["notes"])
else:
    st.info("👈 Upload a model and dataset to get started.")
