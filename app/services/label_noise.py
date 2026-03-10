import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from app.services.preprocessing import prepare_dataset_for_model, detect_target_type

def detect_label_noise(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        return None
    
    # Sampling for expensive ML modules
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    X, y, _ = prepare_dataset_for_model(df, target_column)
    if len(np.unique(y)) < 2:
        return {"suspicious_count": 0}
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)
    pred = rf.predict(X_test)
    classes = rf.classes_
    idx_map = {c: i for i, c in enumerate(classes)}
    high_conf_mismatch = 0
    for i in range(len(y_test)):
        p = float(np.max(proba[i]))
        if p >= 0.9 and pred[i] != y_test[i]:
            high_conf_mismatch += 1
    return {"suspicious_count": int(high_conf_mismatch)}

def feature_importance(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        return None
    
    # Sampling for expensive ML modules
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    X, y, _ = prepare_dataset_for_model(df, target_column)
    if len(np.unique(y)) < 2:
        return {"top_features": []}
    task = detect_target_type(df, target_column)
    if task == "classification":
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X.values, y)
    else:
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X.values, y)
    importances = rf.feature_importances_
    agg = {}
    for name, imp in zip(X.columns.tolist(), importances):
        base = name.split("_")[0] if "_" in name else name
        agg[base] = float(agg.get(base, 0.0) + imp)
    top = sorted([{ "feature": k, "importance": v } for k, v in agg.items()], key=lambda x: x["importance"], reverse=True)[:10]
    return {"top_features": top}
