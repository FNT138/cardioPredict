# src/dataset_load.py
import pandas as pd
from ucimlrepo import fetch_ucirepo
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_uci_heart():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    #Convertir taget multiplase (0,1,2,3,4) a binario (0,1)
    y = (y > 0).astype(int)
    
    df = pd.concat([X, y], axis=1)
    df.rename(columns={df.columns[-1]: "target"}, inplace=True)
    print("UCI Heart Disease:", df.shape)
    return df

def load_kaggle_heart_failure():
    # archivo dentro del dataset: "heart.csv"
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "fedesoriano/heart-failure-prediction",
        "heart.csv"
    )
    # en este dataset la columna objetivo es 'DEATH_EVENT' o similar; normalizamos a 'target'
    if 'DEATH_EVENT' in df.columns:
        df = df.rename(columns={'DEATH_EVENT': 'target'})
    # si no hay 'target' explícito, intentar última columna
    if 'target' not in df.columns:
        df = df.rename(columns={df.columns[-1]: 'target'})
    print("Kaggle Heart Failure:", df.shape)
    return df

def load_kaggle_framingham():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "aasheesh200/framingham-heart-study-dataset",
        "framingham.csv"
    )
    if "TenYearCHD" in df.columns:
        df = df.rename(columns={"TenYearCHD": "target"})
    if 'target' not in df.columns and 'TenYearCHD' not in df.columns:
        # fallback: last column
        df = df.rename(columns={df.columns[-1]:'target'})
    print("Kaggle Framingham:", df.shape)
    return df

if __name__ == "__main__":
    df1 = load_uci_heart()
    df2 = load_kaggle_heart_failure()
    df3 = load_kaggle_framingham()
    # opcional: guardar raw csvs localmente para inspección
    df1.to_csv("data/processed/uci_heart_raw.csv", index=False)
    df2.to_csv("data/processed/heart_failure_raw.csv", index=False)
    df3.to_csv("data/processed/framingham_raw.csv", index=False)
    print("Saved raw CSV copies to data/processed/")
