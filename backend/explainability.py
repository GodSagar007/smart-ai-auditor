# backend/explainability.py

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def is_tree_model(model):
    return isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier))


def is_classification_model(model):
    return hasattr(model, "predict_proba")


def explain_with_shap(model, X_sample, num_features=5):
    """Generate SHAP explanations only if the model is tree-based."""
    if not is_tree_model(model):
        print("⚠️ SHAP explanation skipped: SHAP is best suited for tree-based models like RandomForest or XGBoost.")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        for i in range(min(len(X_sample), 5)):
            print(f"\n🔎 SHAP Explanation for Sample #{i+1}:")
            shap.plots.waterfall(shap.Explanation(values=shap_values[1][i], base_values=explainer.expected_value[1], data=X_sample.iloc[i]), max_display=num_features)
            plt.tight_layout()
            plt.show()

        return shap_values

    except Exception as e:
        print(f"⚠️ SHAP explanation failed: {e}")
        return None


def explain_with_lime(model, X_train, X_sample, feature_names=None, class_names=None):
    """Generate LIME explanation only if model supports probability prediction."""
    if not is_classification_model(model):
        print("⚠️ LIME explanation skipped: model must support predict_proba.")
        return None

    try:
        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names or list(X_train.columns),
            class_names=class_names or ["class_0", "class_1"],
            mode="classification"
        )

        exp = explainer.explain_instance(
            data_row=X_sample.iloc[0].values,
            predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
        )

        print("\n🔍 LIME Explanation for Sample #1:")
        try:
            fig = exp.as_pyplot_figure()
            plt.tight_layout()
            plt.show()
        except:
            print(exp.as_list())

        return exp

    except Exception as e:
        print(f"⚠️ LIME explanation failed: {e}")
        return None
        
def explain_with_shap_ver2(model, X_sample, num_features=5):
    import shap
    import matplotlib.pyplot as plt

    try:
        # Check if the model supports shap.Explainer
        if not hasattr(model, "predict"):
            return {"error": "The model must have a 'predict()' method."}

        # Some models work best with TreeExplainer, others with generic
        try:
            explainer = shap.Explainer(model, X_sample)
        except Exception as e:
            return {"error": f"SHAP failed to create an explainer: {e}"}

        # Get shap values
        shap_values = explainer(X_sample)

        figures = []
        for i in range(min(3, len(X_sample))):
            fig = plt.figure()
            shap.plots.waterfall(shap_values[i], max_display=num_features, show=False)
            figures.append(fig)

        return {"figures": figures}

    except Exception as e:
        return {"error": f"SHAP explanation failed: {e}"}


