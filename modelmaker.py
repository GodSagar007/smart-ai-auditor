# Create and save a sample model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier().fit(X, y)
joblib.dump(model, "models/sample_model.pkl")
