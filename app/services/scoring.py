import pandas as pd

def compute_health_score(df: pd.DataFrame, missing, duplicates, imbalance, leakage, label_noise, correlations, outliers=None):
    score = 100.0
    breakdown = {}
    mv = missing["columns"]
    avg_missing = sum([c["missing_pct"] for c in mv]) / len(mv) if mv else 0.0
    miss_pen = min(30.0, avg_missing * 0.5)
    breakdown["Missing Data Risk"] = "low" if avg_missing < 5 else "medium" if avg_missing < 15 else "high"
    score -= miss_pen
    dup_pct = duplicates["duplicate_pct"]
    dup_pen = min(10.0, dup_pct * 0.3)
    breakdown["Duplicate Risk"] = duplicates["risk_level"]
    score -= dup_pen
    if imbalance:
        max_pct = max(imbalance["distribution_pct"].values()) if imbalance["distribution_pct"] else 0.0
        imb_pen = 0.0 if max_pct < 60 else 8.0 if max_pct < 75 else 15.0
        breakdown["Class Imbalance Risk"] = imbalance["risk_level"]
        score -= imb_pen
    else:
        breakdown["Class Imbalance Risk"] = "unknown"
    hc = correlations["high_correlation_pairs"]
    corr_pen = min(15.0, len(hc) * 2.0)
    breakdown["Feature Correlation Risk"] = "low" if len(hc) == 0 else "medium" if len(hc) < 5 else "high"
    score -= corr_pen
    if outliers:
        out_pct = outliers.get("outlier_pct", 0.0)
        out_pen = min(15.0, out_pct * 0.3)
        breakdown["Outlier Risk"] = "low" if out_pct < 1 else "medium" if out_pct < 5 else "high"
        score -= out_pen
    if leakage:
        leak_pen = min(20.0, len(leakage["suspicious_features"]) * 5.0)
        breakdown["Data Leakage Risk"] = "low" if len(leakage["suspicious_features"]) == 0 else "high"
        score -= leak_pen
    else:
        breakdown["Data Leakage Risk"] = "unknown"
    if label_noise:
        ln_pen = min(10.0, label_noise["suspicious_count"] * 0.5)
        breakdown["Label Noise Risk"] = "low" if label_noise["suspicious_count"] == 0 else "medium" if label_noise["suspicious_count"] < 20 else "high"
        score -= ln_pen
    else:
        breakdown["Label Noise Risk"] = "unknown"
    score = max(0.0, score)
    return {"score": round(score, 2), "breakdown": breakdown}
