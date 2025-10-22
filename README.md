# 🫀 cardioPredict

## 📘 Descripción general

El proyecto **cardioPredict** forma parte de la materia **Inteligencia Artificial** de la **Universidad Nacional de Río Negro (UNRN)**. Tiene como objetivo aplicar técnicas de **aprendizaje automático supervisado (supervised learning)** para desarrollar un modelo capaz de **predecir la presencia de enfermedad cardíaca** en pacientes, a partir de datos clínicos reales.

---

## 🎯 Objetivo del proyecto

El propósito es construir un sistema de clasificación binaria que, a partir de variables médicas (edad, presión arterial, colesterol, frecuencia cardíaca, entre otras), determine si un paciente puede o no tener una enfermedad cardíaca.

El proyecto busca:

* Evaluar el rendimiento de distintos algoritmos de machine learning (Logistic Regression, Random Forest, XGBoost).
* Comparar resultados entre varios conjuntos de datos relacionados con enfermedades del corazón.
* Generar un pipeline reproducible de entrenamiento, evaluación e interpretabilidad.

---

## 🧰 Tecnologías utilizadas

* **Lenguaje:** Python 3.10+
* **Librerías principales:**

  * `pandas`, `numpy` → manipulación de datos
  * `scikit-learn` → modelado y métricas
  * `xgboost` → modelo de boosting
  * `joblib` → guardado de modelos
  * `matplotlib`, `shap` → visualización e interpretabilidad
  * `ucimlrepo`, `kagglehub[pandas-datasets]` → carga automática de datasets

---

## 🧩 Datasets utilizados

El proyecto combina tres fuentes de datos públicas:

1. **UCI Heart Disease Dataset** — (303 registros, 14 variables)
2. **Kaggle Heart Failure Dataset** — (918 registros, 12 variables)
3. **Kaggle Framingham Heart Study Dataset** — (4240 registros, 16 variables)

Cada dataset se descarga automáticamente (vía `ucimlrepo` o `kagglehub`) o puede colocarse manualmente en `data/raw_unzipped/`.

---

## 🧱 Estructura del repositorio

```
cardioPredict/
├── src/
│   ├── dataset_load.py       # Descarga y prepara datasets
│   ├── train_models.py       # Entrena los modelos ML
│   ├── predict_sample.py     # Realiza una predicción individual
│   ├── interpret.py          # Interpretabilidad y visualizaciones
│   └── run_shap.py           # Genera gráficos SHAP
│
├── data/
│   ├── raw/                  # Datos originales
│   ├── processed/            # Datos procesados y splits de test
│
├── models/                   # Modelos entrenados (.joblib)
├── reports/                  # Métricas y resultados
├── notebooks/                # Análisis y pruebas interactivas
└── README.md                 # Este archivo
```

---

## ⚙️ Instalación y configuración

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/FNT138/cardioPredict.git
   cd cardioPredict
   ```

2. Crear un entorno virtual e instalar dependencias:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. (Opcional) Configurar `pre-commit` para validación automática del código:

   ```bash
   pre-commit install
   ```

---

## 🧠 Flujo completo de ejecución

### 🔹 1. Carga y preparación de datos

```bash
python src/dataset_load.py
```

Guarda copias procesadas en `data/processed/*.csv` y muestra shapes en consola.

---

### 🔹 2. Entrenamiento de modelos

```bash
python src/train_models.py
```

Este script:

* Entrena Logistic Regression, Random Forest y XGBoost por cada dataset.
* Guarda modelos en `models/{dataset}_{model}.joblib`.
* Guarda particiones de test (`_X_test.csv` / `_y_test.csv`).
* Exporta métricas en `reports/{dataset}_metrics.json` y `reports/all_metrics.json`.

**Ejemplo de ejecución con log:**

```bash
python src/train_models.py 2>&1 | tee run_train.log
```

---

### 🔹 3. Visualización de resultados

Para inspeccionar las métricas:

```bash
type reports/all_metrics.json
```

O bien en Python:

```python
import pandas as pd
pd.read_json('reports/all_metrics.json')
```

---

### 🔹 4. Predicción para un paciente ejemplo

```bash
python src/predict_sample.py 0.5 models/uci_xgb.joblib models/uci_rf.joblib models/uci_logreg.joblib sample.json
```

**Salida esperada:**

```json
{"probability": 0.83, "prediction": 1}
```

---

### 🔹 5. Interpretabilidad (SHAP)

Ejecutar:

```bash
python notebooks/run_shap.py
```

Genera `reports/shap_summary.png` con la importancia de las variables.

---

### 🔹 6. Ejecución en Jupyter / Colab

En Colab, crear un notebook `CardioPredict.ipynb` y en la primera celda instalar dependencias:

```python
!pip install ucimlrepo kagglehub[pandas-datasets] scikit-learn xgboost shap pandas joblib
```

Luego importar y ejecutar los scripts desde `src/` o copiar las funciones principales.

---

## ⚠️ Problemas comunes y soluciones

| Error                                        | Causa                    | Solución                                                           |
| -------------------------------------------- | ------------------------ | ------------------------------------------------------------------ |
| `ValueError: Unsupported file extension: ''` | Ruta sin extensión       | Usar nombre de archivo completo (`heart.csv`)                      |
| `kagglehub` falla                            | Autenticación o descarga | Descargar manualmente los CSV y colocarlos en `data/raw_unzipped/` |
| SHAP muy lento                               | Dataset grande           | Usar `X_test.sample(n=500)`                                        |
| Warning `use_label_encoder`                  | Versión vieja de XGBoost | Ya está corregido (`use_label_encoder=False`)                      |

---

## 🧩 Reproducibilidad y entrega

Para entregar o replicar el trabajo:

* Incluir:

  * `src/` con scripts funcionales
  * `notebooks/` con notebook ejecutado
  * `models/` con modelos `.joblib`
  * `data/processed/` con `_X_test.csv` y `_y_test.csv`
  * `reports/all_metrics.json` y `reports/*.json`
  * `reports/shap_summary.png`
  * `README.md` (este documento)

* Exportar gráficos (matriz de confusión, ROC, SHAP) para presentación.

---

* **Federico Trujillo**
* **Materia:** Inteligencia Artificial — UNRN
* **Año:** 2025

---
