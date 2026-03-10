import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def detect_target_type(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        return None
    t = df[target_column].dtype
    return "regression" if pd.api.types.is_numeric_dtype(t) else "classification"

def prepare_dataset_for_model(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        raise ValueError("Target column not found")
    df = df.copy()
    df = df.dropna(subset=[target_column])
    task = detect_target_type(df, target_column)
    if task == "classification":
        y_raw = df[target_column].astype(str)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        y = df[target_column].astype(float).values
        le = None
    X = df.drop(columns=[target_column])
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    if num_cols:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        if X[c].isna().any():
            mode_val = X[c].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "missing"
            X[c] = X[c].fillna(fill_val)
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)
    return X, y, le
