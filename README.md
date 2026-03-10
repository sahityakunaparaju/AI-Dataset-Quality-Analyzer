# AI Dataset Quality Analyzer

A lightweight ML data diagnostics tool that analyzes datasets and generates a Dataset Health Report with insights, risk detection, and improvement recommendations.

## Overview

Upload a CSV dataset and receive:
- Dataset statistics
- Detected data quality issues
- Risk levels
- Visualization charts
- ML readiness score
- Recommendations

## Architecture

```
Dataset Upload
↓
Data Validation
↓
Data Preprocessing
↓
Quality Analysis Engine
↓
ML Risk Detection
↓
Dataset Health Score Calculation
↓
Visualization Dashboard
↓
Dataset Health Report
```

## Features

- Dataset statistics: rows, columns, dtypes, numeric summary
- Missing values detection with risk and recommendations
- Duplicate records detection
- Class imbalance analysis
- Feature correlation analysis with high-correlation pairs
- Data leakage detection
- Label noise detection via baseline model
- Feature importance analysis
- Data drift detection between dataset versions
- Dataset version comparison
- ML readiness score

## Tech Stack

- Python
- FastAPI
- Pandas, NumPy
- Scikit-learn
- SciPy
- Plotly, Seaborn
- Streamlit

## Project Structure

```
dataset-quality-analyzer/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes.py
│   ├── services/
│   │   ├── preprocessing.py
│   │   ├── statistics.py
│   │   ├── missing_values.py
│   │   ├── duplicates.py
│   │   ├── imbalance.py
│   │   ├── correlation_analysis.py
│   │   ├── leakage_detection.py
│   │   ├── label_noise.py
│   │   ├── drift_detection.py
│   │   ├── scoring.py
│   ├── models/
│   │   └── baseline_model.py
│   ├── utils/
│   │   └── helpers.py
├── dashboard/
│   └── streamlit_dashboard.py
├── requirements.txt
└── README.md
```

## Installation

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the API

```
uvicorn app.main:app --reload
```

API endpoints:
- GET /health
- POST /api/analyze (multipart: file, optional target_column)
- POST /api/compare (multipart: baseline_file, new_file)

## Running the Dashboard

```
streamlit run dashboard/streamlit_dashboard.py
```

## Usage

1. Start the API server.
2. Open the dashboard.
3. Upload a dataset CSV.
4. Optionally set the target column.
5. Review the Dataset Health Report visuals and recommendations.

## Screenshots

- Missing value chart
- Class distribution
- Correlation heatmap
- Feature importance bar chart
- Health score summary
