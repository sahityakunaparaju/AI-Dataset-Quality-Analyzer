import pandas as pd
import os
from app.services.drift_detection import detect_data_drift

LATEST_DATASET_CACHE = "data/latest_dataset.csv"

def detect_drift_against_previous_dataset(df: pd.DataFrame):
    """Detects drift between the current dataset and the last uploaded one."""
    if not os.path.exists(LATEST_DATASET_CACHE):
        # Cache current for next run
        df.to_csv(LATEST_DATASET_CACHE, index=False)
        return {"drift_detected": False, "message": "First dataset uploaded, baseline established."}
    
    try:
        baseline_df = pd.read_csv(LATEST_DATASET_CACHE)
        drift_result = detect_data_drift(baseline_df, df)
        
        # Update cache with latest for next comparison
        df.to_csv(LATEST_DATASET_CACHE, index=False)
        
        drift_features = drift_result.get("drift_features", [])
        return {
            "drift_detected": len(drift_features) > 0,
            "drift_features": drift_features
        }
    except Exception as e:
        return {"error": f"Drift monitoring failed: {str(e)}"}
