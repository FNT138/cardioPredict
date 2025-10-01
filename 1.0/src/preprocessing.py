# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline(df):
    # detecta num y cat automáticamente
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    # excluimos target
    if 'target' in num_cols:
        num_cols.remove('target')
    # Hay columnas numéricas que son categóricas (sex, cp, fbs) => conviértelas a categóricas según conocimiento
    maybe_cat = ['sex','cp','fbs','thal','slope','ca']
    for c in maybe_cat:
        if c in num_cols:
            num_cols.remove(c)
            cat_cols.append(c)
    # pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop')
    return preprocessor, num_cols, cat_cols
