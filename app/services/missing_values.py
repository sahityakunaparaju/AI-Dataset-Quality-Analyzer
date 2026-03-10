import pandas as pd

def analyze_missing_values(df: pd.DataFrame):
    total = len(df)
    result = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = float(missing / total * 100) if total > 0 else 0.0
        level = "high" if pct >= 30 else "medium" if pct >= 10 else "low"
        rec = "impute or remove" if pct >= 30 else "impute" if pct >= 10 else "no action"
        result.append({"column": col, "missing_count": missing, "missing_pct": pct, "risk_level": level, "recommendation": rec})
    return {"columns": result}
