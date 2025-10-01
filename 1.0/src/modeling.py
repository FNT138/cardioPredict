# src/modeling.py
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from src.preprocessing import build_preprocessing_pipeline
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def run_training(processed_csv='data/processed/heart_raw_processed.csv'):
    df = pd.read_csv(processed_csv)
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
    pre, _, _ = build_preprocessing_pipeline(df)
    pipe = Pipeline([('pre', pre), ('clf', XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42))])
    pipe.fit(X_train, y_train)
    dump(pipe, 'models/best_model.joblib')
    print("Training finished and model saved.")

if __name__ == "__main__":
    run_training()
