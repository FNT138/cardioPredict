"""
    Construye el preprocesador de datos para el DataFrame dado. 
    Identifica columnas numéricas y categóricas, y crea un ColumnTransformer
    que aplica escalado y codificación según corresponda.
@Author Federico Trujillo
@date 2025-10-22    
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:

    # identificar columnas numéricas y categóricas
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if "target" in num_cols:
        num_cols.remove("target")

    # variables que a menudo son numéricas pero conceptualmente categóricas:
    maybe_cat = ["sex", "cp", "fbs", "thal", "slope", "ca", "restecg", "exang"]
    for c in maybe_cat:
        if c in num_cols:
            num_cols.remove(c)
            if c not in cat_cols:
                cat_cols.append(c)

    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)],
        remainder="drop",
    )
    
    return preprocessor, num_cols, cat_cols
