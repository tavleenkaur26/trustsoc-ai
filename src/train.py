import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_clean_data(raw_dir="data/raw"):
    files = glob.glob(f"{raw_dir}/*.parquet")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    web_attack_labels = ['Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - Sql Injection']
    df['Label'] = df['Label'].apply(lambda x: 'Web Attack' if x in web_attack_labels else x)

    exclude_labels = ['Heartbleed', 'Infiltration']
    df_model = df[~df['Label'].isin(exclude_labels)].copy()
    return df_model

def train_model(df_model, output_dir="outputs"):
    le = LabelEncoder()
    y = le.fit_transform(df_model['Label'])
    X = df_model.drop(columns=['Label'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    bot_idx = list(le.classes_).index('Bot')
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == bot_idx] = 3

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric='mlogloss', n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    joblib.dump(model, f'{output_dir}/final_xgb_model.pkl')
    joblib.dump(le, f'{output_dir}/label_encoder.pkl')
    joblib.dump(X_test, f'{output_dir}/X_test.pkl')
    joblib.dump(y_test, f'{output_dir}/y_test.pkl')

    return model, le, X_train, X_test, y_train, y_test

def load_and_clean_data(raw_dir="data/raw"):
    files = glob.glob(f"{raw_dir}/*.parquet")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df['Label'] = df['Label'].str.replace('Web Attack \ufffd', 'Web Attack -', regex=False)
    df['Label'] = df['Label'].str.strip()

    web_attack_labels = ['Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - Sql Injection']
    df['Label'] = df['Label'].apply(lambda x: 'Web Attack' if x in web_attack_labels else x)
    exclude_labels = ['Heartbleed', 'Infiltration']
    df_model = df[~df['Label'].isin(exclude_labels)].copy()
    return df_model

if __name__ == "__main__":
    df_model = load_and_clean_data()
    model, le, X_train, X_test, y_train, y_test = train_model(df_model)
    print("Training complete. Model and artifacts saved to outputs/")