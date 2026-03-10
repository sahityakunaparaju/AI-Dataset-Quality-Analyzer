import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_outliers(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] == 0:
        return {"zscore_per_feature": [], "isolation_forest_total": 0, "outlier_pct": 0.0, "anomaly_indices": []}
    zscore_counts = []
    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.std(ddof=0) == 0 or len(s) == 0:
            zc = 0
        else:
            z = (s - s.mean()) / s.std(ddof=0)
            zc = int((np.abs(z) > 3).sum())
        zscore_counts.append({"feature": col, "outliers": zc})
    X = numeric.fillna(numeric.median())
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso.fit(X)
    preds = iso.predict(X)
    anomaly_idx = list(np.where(preds == -1)[0])
    total_anomalies = int((preds == -1).sum())
    outlier_pct = float(total_anomalies / len(X) * 100.0) if len(X) > 0 else 0.0
    return {
        "zscore_per_feature": zscore_counts,
        "isolation_forest_total": total_anomalies,
        "outlier_pct": outlier_pct,
        "anomaly_indices": anomaly_idx,
    }
