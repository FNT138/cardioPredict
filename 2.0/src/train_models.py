# train_models.py
# --- Entrenamiento y evaluación en cada dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import joblib

from datasets_load import load_uci_heart, load_kaggle_heart_failure, load_kaggle_framingham
from preprocess import build_preprocessing

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs),
        "confusion": confusion_matrix(y_test, preds).tolist()
    }

def train_on_dataset(df, name="dataset"):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    pre, _, _ = build_preprocessing(df)

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "xgb": XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
    }

    results = {}
    for mname, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        results[mname] = evaluate_model(pipe, X_test, y_test)
        joblib.dump(pipe, f"models/{name}_{mname}.joblib")
        print(f"[{name}] {mname}: {results[mname]}")
    return results

if __name__ == "__main__":
    dfs = {
        "uci": load_uci_heart(),
        "hf": load_kaggle_heart_failure(),
        "framingham": load_kaggle_framingham()
    }
    all_results = {}
    for name, df in dfs.items():
        res = train_on_dataset(df, name)
        all_results[name] = res
    print(all_results)
