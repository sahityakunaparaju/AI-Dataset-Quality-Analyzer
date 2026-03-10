import pandas as pd

def analyze_class_imbalance(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        return None
    vc = df[target_column].astype(str).value_counts(normalize=True, dropna=False) * 100.0
    dist = {str(k): float(v) for k, v in vc.items()}
    max_pct = max(dist.values()) if dist else 0.0
    level = "high" if max_pct >= 80 else "medium" if max_pct >= 65 else "low"
    recs = ["SMOTE oversampling", "class-weighted loss", "collect additional samples"]
    return {"distribution_pct": dist, "risk_level": level, "recommendations": recs}
