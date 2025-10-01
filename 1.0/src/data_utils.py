# src/data_utils.py
import os
import zipfile
import pandas as pd

def unzip_all(raw_dir='data/raw', out_dir='data/raw_unzipped'):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(raw_dir):
        if f.lower().endswith('.zip'):
            path = os.path.join(raw_dir, f)
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(out_dir)
    print("Unzipped all files to", out_dir)

def load_csv_guess(in_dir='data/raw_unzipped'):
    for root, _, files in os.walk(in_dir):
        for fname in files:
            if fname.lower().endswith(('.csv', '.data')):
                path = os.path.join(root, fname)
                print("Loading:", path)
                # si es .data → no tiene header
                if fname.lower().endswith('.data'):
                    df = pd.read_csv(path, header=None)
                    # asignar nombres según UCI Cleveland
                    df.columns = [
                        "age","sex","cp","trestbps","chol","fbs","restecg",
                        "thalach","exang","oldpeak","slope","ca","thal","target"
                    ]
                    return df
                else:
                    return pd.read_csv(path)
    raise FileNotFoundError("No se encontró CSV o DATA en " + in_dir)

