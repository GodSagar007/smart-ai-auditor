# frontend/app.py
import os, sys, pickle, joblib
import streamlit as st
import pandas as pd

# ── allow "backend" imports (sibling folder) ──────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import cloudpickle
except ImportError:
    cloudpickle = None

# ── backend modules ───────────────────────────────────────────────────────
from backend.bias_audit import audit_fairness
from backend.drift_detection import detect_data_drift
from backend.robustness import evaluate_robustness
from backend.explainability import explain_with_shap_ver2, explain_with_lime

# ── Streamlit page setup ─────────────────────────────────────────────────
st.set_page_config(page_title="Smart AI Auditor", layout="wide")
st.title("🧠 Smart AI Auditor")

# ── sidebar: file uploads ────────────────────────────────────────────────
st.sidebar.header("Upload Model & Data")
model_file = st.sidebar.file_uploader("Model (.pkl)", type=["pkl"])
data_file  = st.sidebar.file_uploader("Dataset (.csv)", type=["csv"])
label_col  = st.sidebar.text_input("Label Column (ground truth)", value="label")

# ── universal model loader ───────────────────────────────────────────────
def load_model(fp):
    loaders = [
        ("joblib", joblib.load),
        ("pickle", pickle.load),
        ("cloudpickle", cloudpickle.load if cloudpickle else None),
    ]
    for name, loader in loaders:
        if loader is None:
            continue
        try:
            fp.seek(0)
            return loader(fp)
        except Exception as e:
            st.warning(f"• {name} loader failed: {e}")
    return None

# ── main logic ───────────────────────────────────────────────────────────
if model_file and data_file:
    model = load_model(model_file)
    if model is None:
        st.error("❌ Could not load model (unsupported pickle format).")
        st.stop()

    df = pd.read_csv(data_file)
    if label_col not in df.columns:
        st.error(f"❌ Label column '{label_col}' not found in dataset.")
        st.stop()

    X, y = df.drop(columns=[label_col]), df[label_col]
    st.success("✅ Model & data loaded successfully!")

    audit_choice = st.selectbox("Choose an audit", 
        ["Bias / Fairness", "Explainability (SHAP + LIME)", "Data Drift", "Robustness"])

    # ───────────── Bias / Fairness ──────────────
    if audit_choice == "Bias / Fairness":
        st.subheader("⚖️ Bias & Fairness Audit")
        sensitive_cols = st.sidebar.multiselect("Sensitive attribute(s)", options=X.columns.tolist())
        if not sensitive_cols:
            st.warning("Please select at least one sensitive column.")
        else:
            y_pred = model.predict(X)
            for col in sensitive_cols:
                st.markdown(f"### 🔍 Fairness Audit for `{col}`")
                res = audit_fairness(y_true=y, y_pred=y_pred, sensitive_features=X[col])
                st.write("📊 Accuracy by Group", res["accuracy_by_group"])
                st.write("📊 Selection Rate by Group", res["selection_rate_by_group"])
                st.markdown(f"**Verdict:** {res['fairness_interpretation']['notes']}")
                with st.expander("Interpretation Details"):
                    st.json(res["fairness_interpretation"])
                st.divider()

    # ───────────── Explainability ───────────────
    elif audit_choice == "Explainability (SHAP + LIME)":
        st.subheader("🧠 Model Explainability")
        sample = X.sample(n=min(5, len(X)), random_state=42)

        # ---------- SHAP ----------
        st.markdown("### 🔷 SHAP Explanation")
        shap_res = explain_with_shap_ver2(model, sample)

        if shap_res and "figures" in shap_res:
            st.markdown("""
            **What is SHAP?**  
            SHAP assigns each feature a contribution score for a specific prediction.  
            • 🔴 Red bars push the prediction **up**  
            • 🔵 Blue bars push the prediction **down**  
            """)
            for i, fig in enumerate(shap_res["figures"]):
                with st.expander(f"🧾 SHAP for Sample #{i+1}"):
                    st.pyplot(fig)
                    st.caption("Waterfall plot showing feature contributions.")
            st.success("✅ SHAP explanations generated.")
        else:
            reason = shap_res.get("error") if shap_res else "Unknown reason"
            st.warning(f"⚠️ SHAP failed: {reason}")

        st.divider()

        # ---------- LIME ----------
        st.markdown("### 🟡 LIME Explanation")
        try:
            lime_exp = explain_with_lime(model, X, sample, feature_names=list(X.columns))
            if lime_exp:
                st.markdown("""
                **What is LIME?**  
                LIME fits a simple interpretable model near a single data point.  
                • ➕ Positive bars push the prediction **higher**  
                • ➖ Negative bars push the prediction **lower**  
                """)
                try:
                    fig = lime_exp.as_pyplot_figure()
                    with st.expander("🧾 LIME Explanation for first sample"):
                        st.pyplot(fig)
                        st.caption("Local feature importances for the selected sample.")
                except Exception:
                    st.text(lime_exp.as_list())
                st.success("✅ LIME explanation displayed.")
            else:
                st.warning("⚠️ LIME explanation returned None.")
        except Exception as e:
            st.error(f"❌ LIME error: {e}")

    # ───────────── Data Drift ───────────────
    elif audit_choice == "Data Drift":
        st.subheader("📉 Data Drift Detection")
        ref, cur = X.sample(frac=0.5, random_state=1), X.sample(frac=0.5, random_state=2)
        report = detect_data_drift(ref, cur)
        st.metric("Drift Score", f"{report['drift_score']:.2f}")
        st.write(report["notes"])
        with st.expander("Feature‑level stats"):
            st.json(report["feature_stats"])

    # ───────────── Robustness ───────────────
    elif audit_choice == "Robustness":
        st.subheader("🛡️ Robustness Test")
        res = evaluate_robustness(model, X, y)
        if "error" in res:
            st.error(res["error"])
        else:
            st.metric("Original Accuracy", f"{res['original_accuracy']:.3f}")
            st.metric("Noisy Accuracy", f"{res['noisy_accuracy']:.3f}")
            st.metric("Accuracy Drop", f"{res['accuracy_drop']:.3f}")
            st.write(res["notes"])
else:
    st.info("👈 Upload a model and dataset to start auditing.")
