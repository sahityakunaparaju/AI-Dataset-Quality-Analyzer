from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd
import numpy as np
import io
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from app.services.history_tracking import save_dataset_summary
from app.services.statistics import dataset_statistics
from app.services.missing_values import analyze_missing_values
from app.services.duplicates import analyze_duplicates
from app.services.imbalance import analyze_class_imbalance
from app.services.correlation_analysis import analyze_correlations
from app.services.leakage_detection import detect_leakage
from app.services.label_noise import detect_label_noise, feature_importance
from app.services.drift_detection import detect_data_drift, compare_dataset_versions
from app.services.scoring import compute_health_score
from app.utils.helpers import validate_csv_upload, validate_columns
from app.services.outliers import detect_outliers
from app.services.preprocessing import detect_target_type
from app.services.history_tracking import save_dataset_summary, load_dataset_history
from app.services.drift_monitoring import detect_drift_against_previous_dataset
from app.services.recommendations import generate_recommendations
from app.services.baseline_model import train_baseline_model

router = APIRouter()
history_store = []

# ----------------------------
# Convert numpy types to JSON-safe Python types
# ----------------------------
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


# ----------------------------
# Read CSV safely
# ----------------------------
async def read_csv(file: UploadFile) -> pd.DataFrame:
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        # Reset file pointer for potential repeated access if needed
        await file.seek(0)
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")


# ----------------------------
# DATASET ANALYSIS ENDPOINT
# ----------------------------
@router.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
):

    validate_csv_upload(file)

    df = await read_csv(file)

    validate_columns(df)

    if target_column and target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")

    # ---------- PARALLEL ANALYSIS ----------
    with ThreadPoolExecutor() as executor:
        # Submit basic diagnostics
        stats_future = executor.submit(dataset_statistics, df)
        missing_future = executor.submit(analyze_missing_values, df)
        duplicates_future = executor.submit(analyze_duplicates, df)
        correlations_future = executor.submit(analyze_correlations, df)
        outliers_future = executor.submit(detect_outliers, df)

        # Get basic results with error handling
        try:
            stats = stats_future.result()
        except Exception as e:
            stats = {"error": str(e)}

        try:
            missing = missing_future.result()
        except Exception as e:
            missing = {"error": str(e)}

        try:
            duplicates = duplicates_future.result()
        except Exception as e:
            duplicates = {"error": str(e)}

        try:
            correlations = correlations_future.result()
        except Exception as e:
            correlations = {"error": str(e)}

        try:
            outliers = outliers_future.result()
        except Exception as e:
            outliers = {
                "error": str(e),
                "zscore_per_feature": [],
                "isolation_forest_total": 0,
                "outlier_pct": 0.0,
                "anomaly_indices": [],
            }

    # ---------- TARGET BASED ANALYSIS ----------
    imbalance = None
    leakage = None
    label_noise_info = None
    importance = None

    if target_column:
        with ThreadPoolExecutor() as executor:
            # Submit target-based tasks
            imbalance_future = executor.submit(analyze_class_imbalance, df, target_column)
            leakage_future = executor.submit(detect_leakage, df, target_column)
            
            is_classification = detect_target_type(df, target_column) == "classification"
            label_noise_future = None
            if is_classification:
                label_noise_future = executor.submit(detect_label_noise, df, target_column)
            
            importance_future = executor.submit(feature_importance, df, target_column)

            # Retrieve results
            try:
                imbalance = imbalance_future.result()
            except Exception as e:
                imbalance = {"error": str(e)}

            try:
                leakage = leakage_future.result()
            except Exception as e:
                leakage = {"error": str(e)}

            if label_noise_future:
                try:
                    label_noise_info = label_noise_future.result()
                except Exception as e:
                    label_noise_info = {"error": str(e)}

            try:
                importance = importance_future.result()
            except Exception as e:
                importance = {"error": str(e)}

    # ---------- HEALTH SCORE ----------
    try:
        score = compute_health_score(
            df=df,
            missing=missing,
            duplicates=duplicates,
            imbalance=imbalance,
            leakage=leakage,
            label_noise=label_noise_info,
            correlations=correlations,
            outliers=outliers if isinstance(outliers, dict) else None,
        )
    except Exception as e:
        score = {"error": str(e), "score": 0, "breakdown": {}}

    # ---------- NEW FEATURES (MONITORING & RECOMMENDATIONS) ----------
    try:
        recs = generate_recommendations(missing, correlations, outliers, imbalance)
    except Exception as e:
        recs = {"error": str(e), "recommendations": []}

    try:
        baseline_model = train_baseline_model(df, target_column) if target_column else None
    except Exception as e:
        baseline_model = {"error": str(e)}

    try:
        drift_monitoring = detect_drift_against_previous_dataset(df)
    except Exception as e:
        drift_monitoring = {"error": str(e)}

    # ---------- HISTORY TRACKING ----------
    try:
        summary = {
            "dataset_name": file.filename,
            "rows": stats.get("rows", 0),
            "columns": stats.get("columns", 0),
            "numeric_features": stats.get("numeric_feature_count", 0),
            "categorical_features": stats.get("categorical_feature_count", 0),
            "health_score": score.get("score", 0),
        }
        save_dataset_summary(summary)
    except Exception:
        pass

    result = {
        "statistics": stats,
        "missing_values": missing,
        "duplicates": duplicates,
        "class_imbalance": imbalance,
        "correlations": correlations,
        "outliers": outliers,
        "leakage": leakage,
        "label_noise": label_noise_info,
        "feature_importance": importance,
        "health_score": score,
        "baseline_model": baseline_model,
        "drift_monitoring": drift_monitoring,
        "recommendations": recs,
    }
    summary = {
    "dataset_name": file.filename,
    "rows": len(df),
    "columns": df.shape[1],
    "numeric_features": stats.get("numeric_feature_count", 0),
    "categorical_features": stats.get("categorical_feature_count", 0),
    "health_score": score.get("score", 0)
    }

    save_dataset_summary(summary)

    return convert_numpy(result)


# ----------------------------
# DATASET HISTORY ENDPOINT
# ----------------------------
@router.get("/history")
async def get_history():
    return load_dataset_history()


# ----------------------------
# DATASET COMPARISON ENDPOINT
# ----------------------------
@router.post("/compare")
async def compare_datasets(
    baseline_file: UploadFile = File(...),
    new_file: UploadFile = File(...),
):

    validate_csv_upload(baseline_file)
    validate_csv_upload(new_file)

    df_base = await read_csv(baseline_file)
    df_new = await read_csv(new_file)

    validate_columns(df_base)
    validate_columns(df_new)

    drift = detect_data_drift(df_base, df_new)
    version = compare_dataset_versions(df_base, df_new)

    result = {
        "drift": drift,
        "version_comparison": version,
    }

    return convert_numpy(result)