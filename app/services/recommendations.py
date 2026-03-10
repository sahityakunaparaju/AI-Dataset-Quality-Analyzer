def generate_recommendations(missing, correlations, outliers, imbalance):
    """Generates smart recommendations to clean and improve the dataset."""
    recs = []
    
    # Missing values recommendations
    # missing["columns"] is a list of {column, missing_pct, risk_level, ...}
    for col_info in missing.get("columns", []):
        col = col_info.get("column")
        pct = col_info.get("missing_pct", 0)
        
        if 5 <= pct <= 30:
            recs.append(f"Impute missing values in '{col}' using median or mode.")
        elif pct > 30:
            recs.append(f"High missing values in '{col}' ({pct:.1f}%). Recommend dropping this feature.")
            
    # Correlation recommendations
    # correlations["high_correlation_pairs"] is a list of {feature_a, feature_b, correlation}
    for pair in correlations.get("high_correlation_pairs", []):
        f1, f2 = pair.get("feature_a"), pair.get("feature_b")
        recs.append(f"Features '{f1}' and '{f2}' are highly correlated. Recommend removing one to reduce redundancy.")
        
    # Outliers recommendations
    # outliers["outlier_pct"] is the percentage of rows flagged by Isolation Forest
    outlier_pct = outliers.get("outlier_pct", 0)
    if outlier_pct > 10:
        recs.append(f"High outlier percentage detected ({outlier_pct:.1f}%). Investigate potential measurement errors.")
        
    # Class imbalance recommendations
    # imbalance["distribution_pct"] is {class: pct}
    if imbalance and "distribution_pct" in imbalance:
        dist = imbalance["distribution_pct"]
        if dist:
            max_class_pct = max(dist.values())
            if max_class_pct > 70:
                recs.append(f"Class imbalance detected (majority class: {max_class_pct:.1f}%). Recommend SMOTE or resampling.")
                
    return {"recommendations": sorted(list(set(recs)))}
