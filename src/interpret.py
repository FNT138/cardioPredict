"""
Script para interpretar modelos entrenados usando SHAP.
Ajusta según el dataset y modelo que quieras interpretar.

@author: Federico Trujillo
@date: 2025-10-22

"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Cargar modelo entrenado (ejemplo: uci_xgb)
model = joblib.load("models/uci_xgb.joblib")

# Para SHAP necesitamos datos de entrenamiento procesados
df = pd.read_csv("data/processed/heart_raw_processed.csv") if False else None

# Si no guardaste un CSV procesado, podés reusar el X_test que tenías
# en train_models.py (recomiendo guardar X_test.to_csv en esa fase).

# Explicador
explainer = shap.TreeExplainer(model.named_steps["clf"])
# Para usar: X_trans = model.named_steps["pre"].transform(X_test_sample)

print("Ajustar este script según el dataset que se quiera interpretar.")
