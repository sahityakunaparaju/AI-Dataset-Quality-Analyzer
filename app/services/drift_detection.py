import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

def detect_data_drift(baseline: pd.DataFrame, new: pd.DataFrame, alpha: float = 0.05):

    common_cols = [c for c in baseline.columns if c in new.columns]
    drifted = []

    for col in common_cols:

        a = baseline[col].dropna()
        b = new[col].dropna()

        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):

            if len(a) > 0 and len(b) > 0:
                _, p = ks_2samp(a, b)
            else:
                p = 1.0

        else:

            va = a.value_counts()
            vb = b.value_counts()

            cats = sorted(set(va.index).union(set(vb.index)))

            obs = np.array([[va.get(k, 0), vb.get(k, 0)] for k in cats])

            if obs.size == 0:
                p = 1.0
            else:
                try:
                    _, p, _, _ = chi2_contingency(obs)
                except Exception:
                    p = 1.0

        if p < alpha:

            severity = "medium"
            if p < 0.01:
                severity = "high"

            drifted.append({
                "feature": col,
                "p_value": float(p),
                "severity": severity
            })

    return {"drift_features": drifted}


def compare_dataset_versions(baseline: pd.DataFrame, new: pd.DataFrame):

    base_cols = set(baseline.columns)
    new_cols = set(new.columns)

    added = sorted(list(new_cols - base_cols))
    removed = sorted(list(base_cols - new_cols))

    row_diff = int(len(new) - len(baseline))

    return {
        "added_features": added,
        "removed_features": removed,
        "row_count_difference": row_diff
    }