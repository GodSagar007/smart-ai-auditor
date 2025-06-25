# backend/loader.py

import pandas as pd
import joblib
from backend.config import MODEL_PATH, DATA_PATH

def load_model(path=MODEL_PATH):
    try:
        model = joblib.load(path)
        print(f"[✓] Model loaded from: {path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path)
        print(f"[✓] Data loaded from: {path} with shape {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")
