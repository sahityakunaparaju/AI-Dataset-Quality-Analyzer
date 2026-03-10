import pandas as pd
import numpy as np
from app.services.preprocessing import prepare_dataset_for_model

def detect_leakage(df: pd.DataFrame, target_column: str, threshold: float = 0.95):
    if target_column not in df.columns:
        return None
    
    # Sampling for expensive ML modules
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    try:
        X, y, _ = prepare_dataset_for_model(df, target_column)
    except Exception:
        return {"suspicious_features": []}
    features = []
    y_series = pd.Series(y)
    for col in X.columns:
        try:
            c = float(pd.Series(X[col]).corr(y_series))
        except Exception:
            c = 0.0
        if abs(c) >= threshold:
            base = col.split("_")[0] if "_" in col else col
            features.append({"feature": base, "correlation_with_target": abs(c), "risk": "potential leakage"})
    return {"suspicious_features": features}
