""" 
Entrypoint para entrenar modelos en los datasets disponibles.
Carga los datos, aplica preprocesamiento, entrena varios modelos y guarda los resultados.

@Author Federico Trujillo
@date 2025-10-22
"""

import json
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from dataset_load import (
    load_kaggle_framingham,
    load_kaggle_heart_failure,
    load_uci_heart,
)

from preprocess import build_preprocessing

# crear directorios si no existen
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evalúa el modelo en el conjunto de prueba y devuelve métricas de la evaluacion.

    Args:
        model: modelo entrenado
        X_test: características de prueba
        y_test: etiquetas de prueba

    Returns:
        dict: métricas de evaluación
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_test, probs)) if len(set(y_test)) > 1 else None,
        "confusion": confusion_matrix(y_test, preds).tolist(),
    }


def train_on_dataset(df, name="dataset") -> dict:
    """
    Entrena varios modelos en el dataset dado y guarda los resultados.
    Args:
        df: DataFrame con datos y columna 'target'
        name: nombre del dataset (para guardar archivos)
    Returns:
        dict: métricas de evaluación por modelo
    """
    # Normalizar nombre para archivos
    name = name.replace(" ", "_").lower()
    X = df.drop(columns=["target"])
    y = df["target"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # construir preprocesador basado en el df completo (asegura mismas columnas)
    pre, num_cols, cat_cols = build_preprocessing(df)

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "xgb": XGBClassifier(
            eval_metric="logloss",random_state=42
        ),
    }

    results = {}
    for mname, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        # save model
        model_path = f"models/{name}_{mname}.joblib"
        joblib.dump(pipe, model_path)
        print(f"Saved model {model_path}")
        # Evaluate on test
        metrics = evaluate_model(pipe, X_test, y_test)
        results[mname] = metrics

    # save test split for reproducibility
    # Guardamos X_test e y_test para futuras evaluaciones
    X_test.to_csv(f"data/processed/{name}_X_test.csv", index=False)
    y_test.to_csv(f"data/processed/{name}_y_test.csv", index=False)
    # save results json
    with open(f"reports/{name}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{name}] saved X_test, y_test and metrics.")

    return results


if __name__ == "__main__":
    # cargar datasets
    datasets = {
        "uci": load_uci_heart(),
        "heart_failure": load_kaggle_heart_failure(),
        "framingham": load_kaggle_framingham(),
    }
    master = {}
    # entrenar en cada dataset
    for name, df in datasets.items():
        try:
            res = train_on_dataset(df, name)
            master[name] = res
        except Exception as e:
            print(f"Error training on {name}: {e}")
    # guardar resumen global
    with open("reports/all_metrics.json", "w") as f:
        json.dump(master, f, indent=2)
    print(" Entrenamiento finalizado. Todas las métricas en reports/")
