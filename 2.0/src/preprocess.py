# preprocess.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing(df: pd.DataFrame):
    # separar num y cat
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    if "target" in num_cols:
        num_cols.remove("target")

    # algunas variables numéricas son categóricas (ej: sex, cp, fbs, thal en UCI)
    maybe_cat = ["sex","cp","fbs","thal","slope","ca"]
    for c in maybe_cat:
        if c in num_cols:
            num_cols.remove(c)
            cat_cols.append(c)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return pre, num_cols, cat_cols
