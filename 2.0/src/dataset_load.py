# datasets_load.py
import pandas as pd
from ucimlrepo import fetch_ucirepo
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_uci_heart():
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)
    df.rename(columns={df.columns[-1]: "target"}, inplace=True)
    print("UCI Heart Disease:", df.shape)
    return df

def load_kaggle_heart_failure():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "fedesoriano/heart-failure-prediction",
        "heart.csv"    # 👈 archivo dentro del dataset
    )
    df.rename(columns={df.columns[-1]: "target"}, inplace=True)
    print("Kaggle Heart Failure:", df.shape)
    return df

def load_kaggle_framingham():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "aasheesh200/framingham-heart-study-dataset",
        "framingham.csv"   # 👈 archivo dentro del dataset
    )
    if "TenYearCHD" in df.columns:
        df.rename(columns={"TenYearCHD": "target"}, inplace=True)
    print("Kaggle Framingham:", df.shape)
    return df

if __name__ == "__main__":
    df1 = load_uci_heart()
    df2 = load_kaggle_heart_failure()
    df3 = load_kaggle_framingham()
