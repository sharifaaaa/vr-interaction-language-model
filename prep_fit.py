# prep_fit.py
from __future__ import annotations
import pickle
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

ART_DIR = "./artifacts"

def fit_numeric_scaler(train_df: pd.DataFrame, cols: list[str]) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(train_df[cols])
    return sc

def apply_numeric_scaler(df: pd.DataFrame, cols: list[str], sc: StandardScaler) -> pd.DataFrame:
    # ✅ Safe-guard: if there's nothing to transform, return as-is
    if df is None or df.empty:
        return df.copy()
    # (optional) ensure all columns exist
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df.copy()
    out = df.copy()
    out[cols] = sc.transform(out[cols])
    return out

def save_scaler(sc: StandardScaler, path=f"{ART_DIR}/scaler.pkl"):
    import os
    os.makedirs(ART_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(sc, f)

def load_scaler(path=f"{ART_DIR}/scaler.pkl") -> StandardScaler:
    with open(path, "rb") as f:
        return pickle.load(f)



def fit_numeric_scaler(train_df, cols):
    sc = StandardScaler()
    sc.fit(train_df[cols])
    return sc

def apply_numeric_scaler(df, cols, sc: StandardScaler):
    out = df.copy()
    if len(out) and len(cols):
        out[cols] = sc.transform(out[cols])
    return out

def save_numeric_scaler(sc: StandardScaler, path: str):
    joblib.dump(sc, path)

def load_numeric_scaler(path: str) -> StandardScaler:
    return joblib.load(path)
