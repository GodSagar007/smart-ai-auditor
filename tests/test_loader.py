# tests/test_loader.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.loader import load_model, load_data

def test_model_loads():
    model = load_model()
    assert model is not None, "Model failed to load."

def test_data_loads():
    data = load_data()
    assert not data.empty, "Data is empty or failed to load."
    assert "gender" in data.columns, "Missing expected 'gender' column."
