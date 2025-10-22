import joblib
import pandas as pd
import numpy as np
import json
import sys
import os

def predict_with_model(model_path, X, threshold=0.5):
    """
    Predice probabilidades y clasificaciones con un modelo dado.
    X: pd.DataFrame con todas las features necesarias.
    threshold: umbral para convertir probabilidad a predicción binaria.
    """
    pipe = joblib.load(model_path)
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    results = [{"probability": float(p), "prediction": int(d)} for p, d in zip(proba, pred)]
    return results

def load_sample(sample_input):
    """Carga el JSON de un archivo o un string."""
    if os.path.exists(sample_input):
        with open(sample_input, "r") as f:
            sample_data = json.load(f)
    else:
        sample_data = json.loads(sample_input)
    # Si es un dict, convertirlo en lista de uno
    if isinstance(sample_data, dict):
        sample_data = [sample_data]
    return pd.DataFrame(sample_data)

if __name__ == "__main__":
    # Uso: python predict_sample.py threshold modelo1 modelo2 ... sample.json
    if len(sys.argv) < 4:
        print("Usage: python predict_sample.py <threshold> <model1> <model2> ... <sample.json>")
        sys.exit(1)

    threshold = float(sys.argv[1])
    model_paths = sys.argv[2:-1]
    sample_input = sys.argv[-1]

    X = load_sample(sample_input)

    final_results = {}
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        try:
            results = predict_with_model(model_path, X, threshold)
            final_results[model_name] = results
        except ValueError as e:
            final_results[model_name] = f"Error: {e}"

    print(json.dumps(final_results, indent=2))
