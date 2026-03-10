import pandas as pd

def analyze_duplicates(df: pd.DataFrame):
    total = len(df)
    dup_mask = df.duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    pct = float(dup_count / total * 100) if total > 0 else 0.0
    level = "high" if pct >= 20 else "medium" if pct >= 5 else "low"
    rec = "remove duplicates"
    return {"duplicate_count": dup_count, "duplicate_pct": pct, "risk_level": level, "recommendation": rec}
