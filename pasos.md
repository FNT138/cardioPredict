
3.1 Ejecutar carga de datasets (opcional, para guardar copias)

Desde la raíz del proyecto:

python .\src\dataset_load.py


Salida esperada: mensajes con shapes y guardado de csv en data/processed/*.csv.

3.2 Entrenamiento completo (usa los 3 datasets)
python .\src\train_models.py


Qué hace:

entrena 3 modelos (logreg, rf, xgb) por cada dataset,

guarda modelos en models/{dataset}_{model}.joblib,

guarda data/processed/{dataset}_X_test.csv y ..._y_test.csv,

guarda métricas en reports/{dataset}_metrics.json y reports/all_metrics.json.

Si querés guardar log de consola:
Windows PowerShell:

python .\src\train_models.py 2>&1 | Tee-Object -FilePath run_train.log


Linux/macOS:

python3 src/train_models.py 2>&1 | tee run_train.log

3.3 Ver métricas guardadas

Abre reports/all_metrics.json o reports/{dataset}_metrics.json. Ejemplo:

type .\reports\all_metrics.json


(o con un editor / Jupyter)

3.4 Probar inferencia con un paciente ejemplo

Ejemplo (ajusta campos según dataset que uses — usa las columnas reales del dataset):

python .\src\predict_sample.py 0.5 models/uci_xgb.joblib models/uci_rf.joblib models/uci_logreg.joblib sample.json


Salida: JSON con probability y prediction (0/1).

4) Interpretabilidad (SHAP) — ejecutar desde notebook o script

Hecho simple: carga el modelo XGBoost y el X_test guardado, aplica SHAP TreeExplainer y guarda la figura.

Copia el siguiente script como src/run_shap.py:

# src/run_shap.py
import joblib, pandas as pd, shap, matplotlib.pyplot as plt, numpy as np
model = joblib.load("models/uci_xgb.joblib")   # ajusta al modelo que quieras
X_test = pd.read_csv("data/processed/uci_X_test.csv")  # ajustar nombre
# transformar X_test con pipeline
pre = model.named_steps['pre']
clf = model.named_steps['clf']
X_trans = pre.transform(X_test)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_trans)
# Si X_trans es numpy array, convertimos a DataFrame para labels
try:
    feature_names = []
    # intentar obtener feature names si ColumnTransformer ofrece get_feature_names_out
    feature_names = pre.get_feature_names_out()
except Exception:
    feature_names = [f"f{i}" for i in range(X_trans.shape[1])]
shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=150)
print("Saved reports/shap_summary.png")


Ejecuta:

python .\src\run_shap.py


Nota: si pre.get_feature_names_out() falla, SHAP seguirá funcionando con nombres generados.

5) Ejecutar todo en Jupyter / Colab (opción recomendada para el TP)

Si trabajás en Jupyter/Colab, crea un notebook CardioPredict.ipynb y en la primera celda instala dependencias (sólo en Colab):

!pip install ucimlrepo kagglehub[pandas-datasets] scikit-learn xgboost shap pandas joblib


Luego copia el flujo: carga (dataset_load.py funciones), ejecutar train_on_dataset (puedes importar desde src.train_models o replicar las celdas) y visualizar resultados (usar pd.read_json('reports/all_metrics.json') para mostrar tablas).

En Colab con Kaggle: si kagglehub da problemas, sube manualmente los CSV al entorno y carga con pd.read_csv.

6) Errores comunes y soluciones rápidas

ValueError: Unsupported file extension: '' → ya lo corregimos: usar dataset_load() con nombre de archivo (p. ej. "heart.csv").

Problemas con kagglehub (autenticación/descarga) → alternativa:

Descargá manualmente el .csv desde Kaggle (botón Download) y subilo a data/raw_unzipped/ y usa pd.read_csv("data/raw_unzipped/heart.csv").

SHAP muy lento / memoria → usar sample() de X_test (por ejemplo X_test.sample(n=500, random_state=42)).

XGBoost error use_label_encoder → ya seteado use_label_encoder=False. Si hay warnings, actualiza xgboost.

7) Reproducibilidad y entrega

Asegurate de incluir en la entrega:

src/ con scripts.

notebooks/ con Notebook ejecutable (si lo usaste).

models/ con los .joblib.

data/processed/ con *_X_test.csv y *_y_test.csv.

reports/all_metrics.json y reports/*.json.

reports/shap_summary.png.

README.md con pasos para ejecutar (puedes usar este mismo checklist).

Para el PPT: exportá gráficos (matrices de confusión, ROC, SHAP) desde los notebooks y pegálos en la plantilla.

8) ¿Querés que haga ahora alguna cosa concreta por vos?

Elijo una de las siguientes acciones y la hago en texto para que pegues en tu proyecto:

Generar el notebook CardioPredict.ipynb completo (celdas listas para Colab/Jupyter).

Entregar un README.md listo y detallado con todos los comandos (listo para subir).

Crear un script Streamlit para demo rápida (subir CSV y predecir).

Decime cuál querés y lo genero en la siguiente respuesta. Si preferís, te paso también las modificaciones exactas para tu train_models.py actual en lugar de reemplazar archivos (si querés mantener histórico).
