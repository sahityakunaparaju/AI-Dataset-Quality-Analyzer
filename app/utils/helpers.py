import pandas as pd
from fastapi import UploadFile, HTTPException

def validate_csv_upload(file: UploadFile, max_size_mb: int = 50):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

def validate_columns(df: pd.DataFrame):
    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Dataset has no columns")
    if df.columns.duplicated().any():
        raise HTTPException(status_code=400, detail="Duplicate column names detected")
