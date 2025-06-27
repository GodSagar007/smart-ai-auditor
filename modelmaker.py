# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load your preprocessed CSV
df = pd.read_csv("data/test_data.csv")

# Separate features and label
X = df.drop("approved", axis=1)
y = df["approved"]

# Train-test split (optional, for real workflows)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/sample_model.pkl")

print("✅ Model trained and saved to models/sample_model.pkl")
