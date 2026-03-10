import pandas as pd

def dataset_statistics(df: pd.DataFrame):
    rows, cols = df.shape
    dtypes = {c: str(dt) for c, dt in df.dtypes.items()}
    numeric_cols = df.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    numeric_summary = df[numeric_cols].describe().to_dict() if numeric_cols else {}
    return {
        "rows": int(rows),
        "columns": int(cols),
        "dtypes": dtypes,
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "datetime_feature_count": len(datetime_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "numeric_summary": numeric_summary,
    }
