import pandas as pd
import numpy as np

def analyze_correlations(df: pd.DataFrame, threshold: float = 0.9):
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] == 0:
        return {"matrix": {}, "high_correlation_pairs": []}
    corr = numeric.corr(numeric_only=True)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(abs(corr.iloc[i, j]))
            if val >= threshold:
                pairs.append({"feature_a": cols[i], "feature_b": cols[j], "correlation": val})
    return {"matrix": corr.to_dict(), "high_correlation_pairs": pairs}
