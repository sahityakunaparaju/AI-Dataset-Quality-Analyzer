import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

from app.services.preprocessing import prepare_dataset_for_model, detect_target_type


def train_baseline_model(df: pd.DataFrame, target_column: str):
    """Trains a baseline ML model (Classification/Regression) and returns metrics."""

    if target_column not in df.columns:
        return None

    # Sample large datasets for ML-heavy tasks
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    try:
        # Prepare dataset
        X, y, le = prepare_dataset_for_model(df, target_column)

        task = detect_target_type(df, target_column)

        # Stratify only for classification
        stratify = y if task == "classification" and len(np.unique(y)) > 1 else None

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify
        )

        # -----------------------------
        # Classification Model
        # -----------------------------
        if task == "classification":

            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            metrics = {
                "model_type": "classification",
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }

        # -----------------------------
        # Regression Model
        # -----------------------------
        else:

            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            metrics = {
                "model_type": "regression",
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2_score": float(r2_score(y_test, y_pred))
            }

        # -----------------------------
        # Feature Importance
        # -----------------------------

        importances = model.feature_importances_

        # Safe feature name extraction
        if hasattr(X, "columns"):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        agg = {}

        for name, imp in zip(feature_names, importances):
            base = name.split("_")[0] if "_" in name else name
            agg[base] = float(agg.get(base, 0.0) + imp)

        top_features = sorted(
            [{"feature": k, "importance": v} for k, v in agg.items()],
            key=lambda x: x["importance"],
            reverse=True
        )[:10]

        metrics["top_features"] = top_features

        return metrics

    except Exception as e:
        return {"error": f"Baseline model training failed: {str(e)}"}